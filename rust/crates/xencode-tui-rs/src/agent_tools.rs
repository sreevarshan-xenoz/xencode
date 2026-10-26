//! Executes the tool calls the model requests in the chat loop:
//! background tasks (Milestone D) and repo insights (Milestone F, F3-02).
//!
//! The [`TaskManager`](xencode_core_rs::TaskManager) lives in an
//! `Arc<tokio::sync::Mutex>` on `App` so the chat loop, later CLI commands
//! and the D2 panel share one registry. Results come back as plain text —
//! readable to the model and cheap to echo into chat.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use xencode_core_rs::{TaskError, TaskManager, TaskRecord};
use xencode_providers_rs::ToolCall;

// How many assistant→tool→assistant rounds one user turn may take before tools
// stop being offered and the model must answer in prose: the `agent_max_rounds`
// config key (default 16, clamped to 1..=64 by the loop). The wording that
// teaches it lives with the tool vocabulary, at
// `xencode_context_rs::prompts::TOOLS`.

// ── Permission policy (I1-01) ───────────────────────────────────────────
// One source of truth for "may the agent run this call?". The chat loop
// (and later the approval overlay) asks `classify`; nothing else decides.

/// Named values for the `agent_approval` config key (Settings row + CLI).
pub const APPROVAL_MODE_NAMES: &[&str] = &["ask", "edit-allow", "all-allow", "plan", "autonomous"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Mutating and shell tools prompt; read-only tools run freely.
    Ask,
    /// File edits are auto-approved; shell still prompts.
    EditAllow,
    /// Everything except hard-denied paths is auto-approved.
    AllAllow,
    /// Read-only, enforced by the gate: an edit or shell call is denied, not
    /// merely prompted, so a plan turn cannot write even on a stale grant.
    Plan,
    /// The whole task runs without a human: reads, edits and shell are free,
    /// but anything reaching a stranger's MCP server or off the machine is
    /// denied rather than asked, because there is nobody to answer the prompt.
    Autonomous,
}

impl ApprovalMode {
    /// Unknown values fall back to the strictest mode (theme/layout precedent).
    pub fn parse(name: &str) -> Self {
        match name {
            "edit-allow" => Self::EditAllow,
            "all-allow" => Self::AllAllow,
            "plan" => Self::Plan,
            "autonomous" => Self::Autonomous,
            _ => Self::Ask,
        }
    }

    /// The word `agent_approval` holds for this mode; `parse` reads it back.
    pub fn config_name(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::EditAllow => "edit-allow",
            Self::AllAllow => "all-allow",
            Self::Plan => "plan",
            Self::Autonomous => "autonomous",
        }
    }
}

/// What a tool touches; drives both the mode decision and session grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    ReadOnly,
    Edit,
    Shell,
    /// A tool belonging to an external MCP server: we cannot preview its
    /// effect, cannot checkpoint it, and cannot undo it.
    External,
    /// A request out to an address the model chose. Its own class rather than a
    /// capability hanging off `Shell`, because the two cannot share a
    /// permission: "always allow commands" is a statement about this machine,
    /// and nothing about this machine implies consent to fetch a page nobody
    /// named. `Network` is the one class the session grant refuses to buy off.
    Network,
}

impl ToolClass {
    /// The words the approval overlay shows for this class. One definition,
    /// because the overlay and the run ledger (QTR-5) describe the same call and
    /// must not describe it two ways.
    pub fn overlay_label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Edit => "file change",
            Self::Shell => "shell command",
            Self::External => "external tool",
            Self::Network => "network request",
        }
    }

    /// The class an approval prompt names in words, read back on a window
    /// that only received the words. Unknown words are treated as the most
    /// guarded class, so a window never shows a prompt as safer than it is.
    pub fn from_overlay_label(label: &str) -> Self {
        [
            Self::ReadOnly,
            Self::Edit,
            Self::Shell,
            Self::External,
            Self::Network,
        ]
        .into_iter()
        .find(|class| class.overlay_label() == label)
        .unwrap_or(Self::Network)
    }
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
/// already size-capped by [`approval_preview`], and `draft` holds the file
/// bytes that diff was worked out from.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool: String,
    pub class: ToolClass,
    pub summary: String,
    pub preview: String,
    pub draft: ApprovalDraft,
}

impl ApprovalRequest {
    pub fn class_label(&self) -> &'static str {
        self.class.overlay_label()
    }
}

/// The file contents the shown preview was computed from.
///
/// `Approved` is consent to *that* change. A file that moves while the prompt
/// is open means the executor's re-read would write something the person never
/// saw, so the gate checks these bytes again before spending the approval —
/// see [`ApprovalDraft::stale_paths`]. A tool whose preview reads no file (a
/// command line, a URL) holds nothing here, and there is nothing to re-check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalDraft {
    /// `(absolute path, workspace-relative path, hash of its bytes then)`,
    /// with `None` for a file the preview found absent — a new file.
    files: Vec<(PathBuf, String, Option<u64>)>,
}

impl ApprovalDraft {
    /// Fingerprint these files as they stand at this moment. A file that is
    /// absent hashes to `None`, which is how "this would create it" stays
    /// distinct from "this would change it".
    pub(crate) fn bind(files: &[(PathBuf, String)]) -> Self {
        ApprovalDraft {
            files: files
                .iter()
                .map(|(full, display)| (full.clone(), display.clone(), bytes_hash(full, display)))
                .collect(),
        }
    }

    /// The bound files whose bytes differ now, by workspace-relative path.
    /// Empty means every file still holds what the person was shown.
    pub fn stale_paths(&self) -> Vec<String> {
        self.files
            .iter()
            .filter(|(full, display, then)| bytes_hash(full, display).ne(then))
            .map(|(_, display, _)| display.clone())
            .collect()
    }

    /// True when nothing the preview read has moved, including when it read
    /// no file at all.
    pub fn is_current(&self) -> bool {
        self.stale_paths().is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// Hash of a file's bytes, or `None` if it does not exist. Only ever compared
/// against another hash taken in the same process a moment earlier, so it
/// needs no strength beyond saying "these are not the same bytes".
fn bytes_hash(full: &Path, display: &str) -> Option<u64> {
    let data = std::fs::read(full).ok()?;
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // The path is hashed with the bytes: an edit approved for `src/a.rs` must
    // not pass its re-check because the same content turned up under a
    // different name.
    display.hash(&mut hasher);
    data.hash(&mut hasher);
    Some(hasher.finish())
}

/// The files the overlay's diff was worked out from, for this call. Every call
/// whose preview is built from file contents is bound here — a single-file
/// write or edit, and the multi-file `ast_edit`/`rename`/`codemod` plans, whose
/// preview is a real diff per file. A preview that comes from the arguments
/// alone (a command line, a URL) binds nothing, so it cannot go stale under the
/// person answering it.
pub fn approval_draft(root: &Path, call: &ToolCall) -> ApprovalDraft {
    approval_shown(root, call).draft
}

/// How many times one call may be re-shown because the file kept moving while
/// the person was answering. Past that, the write is refused: a file being
/// edited faster than it can be reviewed is not a reason to stop reviewing it.
const MAX_DRAFT_REVIEWS: usize = 2;

/// How long the approval preview waits for ast-grep. Shorter than the
/// executor's budget on purpose: a preview that blocks the prompt is worse than
/// one that says the check timed out, and the executor re-runs the same planner
/// with the full allowance before anything is written.
const AST_PREVIEW_TIMEOUT_SECS: u64 = 10;

/// What a tool call touches. Unknown tools count as Shell — the executor
/// errors on them anyway, but they are never silently treated as read-only.
pub fn tool_class(tool: &str) -> ToolClass {
    // A server we did not write, whose effects we cannot preview or undo.
    if crate::mcp::is_mcp_tool(tool) {
        return ToolClass::External;
    }
    match tool {
        "background_poll" | "repo_advise" | "what_breaks" | "find_refs" | "callers"
        | "read_file" | "list_dir" | "search_files" | "read_docs" | "lookup_advisory"
        | "load_skill" | "update_plan" | "ask_user" => ToolClass::ReadOnly,
        "write_file" | "edit_file" | "edit_symbol" | "ast_edit" | "codemod" | "rename" => {
            ToolClass::Edit
        }
        // EV-6: a note is a write — into `.xencode/notes.md`, one line at a time,
        // with no path the model can choose. It is the same class as `write_file`
        // because it is the same kind of act, and plan mode refusing it is the
        // point: a plan that cannot read the workspace also cannot leave marks in
        // the pad that every later turn then reads.
        "write_note" => ToolClass::Edit,
        // `reproduce_bug` runs a command, so it is a shell call by class: it
        // costs whatever `run_command` costs in this mode, never less. Naming it
        // explicitly keeps it out of the unknown-tool fallback while saying why.
        "reproduce_bug" => ToolClass::Shell,
        // CI-7: builds and runs a test binary under a debugger — it runs the
        // project's code, so it costs what running a command costs.
        "debug_test" => ToolClass::Shell,
        // RS-1: the address comes from the model, so the class is the trip
        // itself rather than anything it reads or writes.
        "web_fetch" => ToolClass::Network,
        // RS-2: same reasoning with a different destination. The engine is named
        // in the config, but the *question* comes from the model and leaves this
        // machine, and a search is a trip, not a read.
        "web_search" => ToolClass::Network,
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
/// workspace and outside the forbidden zones (`.git/`, xencode's own
/// directories).
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
    // Every directory xencode keeps in the person's home, not only the settings
    // one: the records, the cache and the downloaded weights are not there any
    // more, and a tool that could write into any of them could delete the audit
    // trail or replace a model file the next turn reads.
    if xencode_config_rs::paths::is_internal(&joined) {
        return false;
    }
    true
}

/// The policy decision for one call. `granted` lists classes the user
/// approved for the whole session at an earlier prompt.
/// What a tool can do, in the gate's own vocabulary (CAP-1). `classify`
/// decides through these, so SE-4's trifecta check and RS-1's network tools
/// read the same words the gate enforced — not a second taxonomy that can
/// drift from the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    FilesystemRead,
    FilesystemWrite,
    ShellExecute,
    NetworkRequest,
    ExternalMcp,
}

impl Capability {
    /// The word the gate, the broker's refusals and the plan all use.
    pub fn name(self) -> &'static str {
        match self {
            Self::FilesystemRead => "filesystem.read",
            Self::FilesystemWrite => "filesystem.write",
            Self::ShellExecute => "shell.execute",
            Self::NetworkRequest => "network.request",
            Self::ExternalMcp => "external.mcp",
        }
    }
}

/// Every capability a tool may exercise. Derived from the tool's class, with
/// one rule the class cannot carry: anything from an MCP server is a
/// stranger's code, whatever its shape claims.
///
/// `network.request` had no mapping when CAP-1 landed: no built-in tool was a
/// pure network tool (`read_docs` fetches only through its own consent flag
/// and stays a read here), and the `sh -c` strings whose network use would
/// need prefix matching are SE-7's kernel job by the item's own trap, not the
/// gate's. RS-1's `web_fetch` is the tool that arrives onto that existing
/// table row — the gate's decision was written before the tool existed, which
/// is the "for free" the plan promised.
pub fn tool_capabilities(tool: &str) -> Vec<Capability> {
    if crate::mcp::is_mcp_tool(tool) {
        return vec![Capability::ExternalMcp];
    }
    match tool_class(tool) {
        ToolClass::ReadOnly => vec![Capability::FilesystemRead],
        ToolClass::Edit => vec![Capability::FilesystemWrite],
        ToolClass::Shell => vec![Capability::ShellExecute],
        ToolClass::Network => vec![Capability::NetworkRequest],
        // Unreachable today — the MCP check above owns every external name —
        // and kept so a future external class cannot fall through silently.
        ToolClass::External => vec![Capability::ExternalMcp],
    }
}

/// What one capability costs in one mode: free, or a prompt. `network.request`
/// asks in every mode even though nothing maps to it yet — fail closed, so
/// when RS-1's tools arrive they inherit the strict row, and widening it is
/// RS-1's decision to make with its own tests, not a side effect found later.
fn capability_gate(capability: Capability, mode: ApprovalMode) -> Permission {
    match mode {
        ApprovalMode::Ask => match capability {
            Capability::FilesystemRead => Permission::Allow,
            _ => Permission::Ask,
        },
        ApprovalMode::EditAllow => match capability {
            Capability::FilesystemRead | Capability::FilesystemWrite => Permission::Allow,
            _ => Permission::Ask,
        },
        // Everything is free except a stranger's server or the network,
        // which always ask: "all-allow" never meant "anyone's code" and
        // never meant "anywhere off this machine".
        ApprovalMode::AllAllow => match capability {
            Capability::ExternalMcp | Capability::NetworkRequest => Permission::Ask,
            _ => Permission::Allow,
        },
        // PLAN is read-only, and the word enforced is Deny rather than Ask. A
        // session grant can only shortcut a prompt (`decision == Ask`), never a
        // denial, so "allow edits for this session" clicked while implementing
        // cannot leak into a later plan and let it write. This is the exact
        // failure the industry call "Plan Mode Isn't Read-Only" describes; here
        // the mode cannot be talked out of being read-only.
        ApprovalMode::Plan => match capability {
            Capability::FilesystemRead => Permission::Allow,
            _ => Permission::Deny,
        },
        // AUTONOMOUS does the local work without prompting — reads, edits and
        // shell — but still refuses to reach a stranger's MCP server or leave
        // the machine, denied rather than asked so it can run unattended without
        // hanging on a prompt nobody is there to answer. AllAllow lets those
        // prompt; AUTONOMOUS closes them.
        ApprovalMode::Autonomous => match capability {
            Capability::FilesystemRead | Capability::FilesystemWrite | Capability::ShellExecute => {
                Permission::Allow
            }
            Capability::NetworkRequest | Capability::ExternalMcp => Permission::Deny,
        },
    }
}

pub fn classify(
    root: &Path,
    tool: &str,
    args: &serde_json::Map<String, serde_json::Value>,
    mode: ApprovalMode,
    granted: &[ToolClass],
    tainted: bool,
) -> Permission {
    let external = crate::mcp::is_mcp_tool(tool);
    // Path arguments are hard-denied outside the workspace in every mode:
    // "all-allow" never means "anywhere on disk". That rule is about *our* file
    // tools; a server's own `path` argument means something in the server's
    // filesystem, so refusing it here would break the server rather than
    // protect the workspace. Those calls still always prompt, where the
    // arguments are shown. The one carve-out is read-only and narrow, described
    // by the `CRATE_AWARE_TOOLS` list.
    if !external && escapes_workspace(root, tool, args) {
        return Permission::Deny;
    }
    // Decided through the capability vocabulary, most restrictive wins: a
    // tool is free only where every capability it carries is free. The
    // session-grant shortcut below is unchanged — an "always allow" answer
    // still replaces the prompt for its class.
    let mut decision = Permission::Allow;
    for capability in tool_capabilities(tool) {
        decision = match (decision, capability_gate(capability, mode)) {
            (Permission::Deny, _) | (_, Permission::Deny) => Permission::Deny,
            (Permission::Ask, _) | (_, Permission::Ask) => Permission::Ask,
            _ => Permission::Allow,
        };
    }
    // SE-4, the lethal-trifecta gate: a session that has touched secrets
    // treats every shell call as asking, in every mode, grants
    // notwithstanding. The grant predates the secret read — the risk emerged
    // after the permission was given, so the permission cannot cover it. A
    // one-shot approval at the prompt still runs the call; headless, the
    // prompt's absence denies it as before. Reads, edits and strangers'
    // servers are untouched: reads and edits cannot exfiltrate by
    // themselves, and a stranger's server already always asks.
    let shell = tool_capabilities(tool).contains(&Capability::ShellExecute);
    if tainted && shell && decision != Permission::Deny {
        return Permission::Ask;
    }
    let class = tool_class(tool);
    // Unreachable for a tainted shell: the rule above already returned Ask,
    // so a grant given before the secrets were read cannot cover a call
    // made after.
    //
    // A network request is exempt from the shortcut for the same kind of reason.
    // A grant is one decision reused, and the decision here was about one
    // address; reusing it for the next is not what was agreed. So every page the
    // model asks for is asked of the person, and "always allow" is refused for
    // this class at the prompt rather than accepted and then ignored.
    if decision == Permission::Ask && class != ToolClass::Network && granted.contains(&class) {
        Permission::Allow
    } else {
        decision
    }
}

/// Whether one of the call's `path`/`cwd` arguments lands outside the workspace
/// or in a forbidden zone. This is the guard `classify` turns into a hard deny,
/// lifted out so the headless policy refuses for exactly the same reason the
/// interactive gate does — the two can never drift on what "outside" means.
fn escapes_workspace(
    root: &Path,
    tool: &str,
    args: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    for key in ["path", "cwd"] {
        if let Some(serde_json::Value::String(raw)) = args.get(key) {
            if raw.is_empty() {
                continue;
            }
            // A `crate:` address is a read and nothing else, so it is reachable
            // only as `path` on one of the three read tools. Anything else that
            // escapes the workspace stays refused.
            let reachable = if crate::crate_sources::is_crate_spec(raw) {
                key == "path" && readable_path(root, tool, raw).is_ok()
            } else {
                path_allowed(root, raw)
            };
            if !reachable {
                return true;
            }
        }
    }
    false
}

// ── Headless (non-interactive) permission policy (M-5) ────────────────
// A caller that cannot answer an approval prompt — an external MCP client
// talking to `xencode mcp serve` — must not be able to approve by silence.
// This is a separate decision from `classify`: the interactive gate never
// consults it, and it never widens that gate. It can only refuse further.

/// A tool a headless caller is permitted to reach. Read-only tools are always
/// permitted; a file-changing or shell tool is permitted only when named at
/// launch, and even then a path outside the workspace stays refused.
#[derive(Debug, Clone, Default)]
pub struct HeadlessPolicy {
    /// Tool names the operator allowed when the server started. Fixed for the
    /// life of the process, so a client cannot widen it by asking.
    allowed: Vec<String>,
}

/// The outcome for one headless call, with the reason a refusal gave. Kept
/// separate from [`Permission`] because there is no "ask" here: a call either
/// runs or does not, and a refusal must say why in the caller's own terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Headless {
    Allow,
    /// `reason` is what the MCP client is told, and it names the flag that
    /// would allow the call — a refusal the caller cannot act on is a dead end.
    Refused {
        reason: String,
    },
}

impl HeadlessPolicy {
    pub fn new(allowed: impl IntoIterator<Item = String>) -> Self {
        let mut allowed: Vec<String> = allowed.into_iter().collect();
        allowed.sort();
        allowed.dedup();
        Self { allowed }
    }

    /// Nothing named: the strictest and default setting — read-only and
    /// nothing else, which is the only safe answer for a caller with no way to
    /// approve.
    pub fn read_only() -> Self {
        Self::new(Vec::new())
    }

    pub fn allows(&self, tool: &str) -> bool {
        self.allowed.iter().any(|name| name == tool)
    }

    /// Whether nothing beyond reads was named at launch — the state a server
    /// says so in its handshake, rather than each caller working it out.
    pub fn is_read_only(&self) -> bool {
        self.allowed.is_empty()
    }

    /// The decision for one call. Read-only tools run. A mutating or shell tool
    /// runs only if the operator named it at launch. A `path` or `cwd` argument
    /// that lands outside the workspace is refused whatever was named, because
    /// that boundary is not the operator's to hand to a caller it cannot see.
    /// What a *granted* shell command then does to paths it names itself is not
    /// checked here — the guard reads the arguments, not the command text — so
    /// `--allow run_command` gives the caller a shell, not a jailed one.
    pub fn decide(
        &self,
        root: &Path,
        tool: &str,
        args: &serde_json::Map<String, serde_json::Value>,
    ) -> Headless {
        if escapes_workspace(root, tool, args) {
            return Headless::Refused {
                reason: format!(
                    "`{tool}` was given a path outside this workspace or in a protected zone; \
                     xencode refuses a call whose `path` or `cwd` leaves the directory it was \
                     started in, whatever was allowed at launch"
                ),
            };
        }
        let class = tool_class(tool);
        if class == ToolClass::ReadOnly {
            return Headless::Allow;
        }
        // Named at launch or not, a caller with no one to ask cannot reach the
        // network: the approval every fetch needs has nowhere to go, so the
        // grant below would only promise a run that the interactive gate then
        // refuses. This says which half is missing instead.
        if class == ToolClass::Network {
            return Headless::Refused {
                reason: format!(
                    "`{tool}` sends a request to an address the caller chose, and this \
                     `xencode mcp serve` has no one to approve that trip. Run it in the \
                     TUI, where each fetch is shown and asked."
                ),
            };
        }
        if self.allows(tool) {
            return Headless::Allow;
        }
        Headless::Refused {
            reason: format!(
                "`{tool}` is a {} tool and this `xencode mcp serve` {}. Start it with \
                 `--allow {tool}` to permit this one tool.",
                class_label(class),
                self.launch_words()
            ),
        }
    }

    /// How a refusal describes this launch. Saying "read-only" after a tool was
    /// granted would be a lie a caller acts on, so the grant is named instead.
    fn launch_words(&self) -> String {
        if self.allowed.is_empty() {
            "was started in read-only mode, so nothing that changes files or runs a \
             command is executed"
                .to_string()
        } else {
            format!(
                "was started permitting only {} beyond the three reads, so it is not \
                 executed",
                self.allowed
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// The word for a tool class in a refusal a human reads.
fn class_label(class: ToolClass) -> &'static str {
    match class {
        ToolClass::ReadOnly => "read-only",
        ToolClass::Edit => "file-changing",
        ToolClass::Shell => "shell",
        ToolClass::External => "external",
        ToolClass::Network => "network",
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

/// The three tools that may also read the unpacked source of a dependency this
/// project locked. Deliberately a list rather than "every read-only tool":
/// `what_breaks` and `repo_advise` take a path that has to mean something to
/// *this* project's index, so widening them would answer a question about a
/// crate in the registry and present it as an answer about this workspace.
const CRATE_AWARE_TOOLS: &[&str] = &["read_file", "list_dir", "search_files"];

/// Resolve a model-supplied path to an absolute in-workspace path, returning
/// the hard-deny error string when it escapes (outside the root, `.git/`,
/// config dir). This is the executor-side mirror of `classify`'s path rule:
/// until the approval prompt (I1-03) is wired, the file tools still refuse
/// every out-of-workspace byte.
pub(crate) fn workspace_path(root: &Path, raw: &str) -> Result<(PathBuf, String), String> {
    if raw.trim().is_empty() {
        return Err(err("\"path\" must not be empty"));
    }
    if crate::crate_sources::is_crate_spec(raw) {
        return Err(err(format!(
            "{raw} addresses a dependency's own source, which is read-only here: read_file, \
             list_dir and search_files can read it"
        )));
    }
    if !path_allowed(root, raw) {
        return Err(err(format!(
            "path outside the workspace (or a forbidden directory): {raw}"
        )));
    }
    let full = resolve_path(root, raw);
    Ok((full, raw.trim().to_string()))
}

/// The path rule for a *read*: the workspace, plus — for the tools in
/// [`CRATE_AWARE_TOOLS`] — the unpacked source of a crate the lock file pins.
/// The third arm is the only carve-out, and it is read-only: a write, an edit or
/// a `cwd` still goes through [`workspace_path`].
///
/// Returns the path, the label to show the model, and the crate the path came
/// from when it was not the workspace.
fn readable_path(
    root: &Path,
    tool: &str,
    raw: &str,
) -> Result<(PathBuf, String, Option<crate::crate_sources::CrateSource>), String> {
    if raw.trim().is_empty() {
        return Err(err("\"path\" must not be empty"));
    }
    if CRATE_AWARE_TOOLS.contains(&tool) {
        if crate::crate_sources::is_crate_spec(raw) {
            let (full, source) = crate::crate_sources::resolve_crate_spec(root, raw)?;
            return Ok((full, raw.trim().to_string(), Some(source)));
        }
        let candidate = resolve_path(root, raw);
        if !path_allowed(root, raw) {
            if let Some(source) = crate::crate_sources::crate_source_for_path(root, &candidate) {
                return Ok((candidate, raw.trim().to_string(), Some(source)));
            }
        }
    }
    workspace_path(root, raw).map(|(path, display)| (path, display, None))
}

/// The line that tells the model which crate and version a read came from, and
/// how to ask for it again without spelling out a registry path. Empty for a
/// workspace read, so ordinary output is unchanged.
fn crate_provenance(source: Option<&crate::crate_sources::CrateSource>, path: &Path) -> String {
    let Some(source) = source else {
        return String::new();
    };
    let inside = source.inside(path).unwrap_or_default();
    format!(
        "[{} — read from {}, unpacked by cargo]\n",
        source.label(),
        source.spec_for(&inside)
    )
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
    // The whole diff is counted before anything is cut, so a change too big to
    // read in a terminal pane still says how big it is: the person is deciding
    // on a summary they can weigh, not on a truncated window they might take
    // for the lot.
    let shown: Vec<String> = TextDiff::from_lines(old, new)
        .unified_diff()
        .iter_hunks()
        .flat_map(|hunk| {
            hunk.to_string()
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect();
    let added = shown.iter().filter(|l| l.starts_with('+')).count();
    let removed = shown.iter().filter(|l| l.starts_with('-')).count();
    let mut out: String = shown
        .iter()
        .take(DIFF_MAX_LINES)
        .map(|l| format!("{l}\n"))
        .collect();
    if shown.len() > DIFF_MAX_LINES {
        out.push_str(&format!(
            "… diff truncated ({} of {} lines shown; {added} added, {removed} removed in all)\n",
            DIFF_MAX_LINES,
            shown.len()
        ));
    }
    out
}

fn tool_read_file(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(raw) = arg_str(args, "path") else {
        return err("read_file needs a string \"path\"");
    };
    let (full, display, source) = match readable_path(root, "read_file", raw) {
        Ok(ok) => ok,
        Err(e) => return e,
    };
    let provenance = crate_provenance(source.as_ref(), &full);
    if full.is_dir() {
        let hint = source
            .as_ref()
            .map(|s| {
                format!(
                    " — ask for a file inside it, for example {} or {}",
                    s.spec_for("Cargo.toml"),
                    s.spec_for("README.md")
                )
            })
            .unwrap_or_default();
        return err(format!("{display} is a directory{hint}"));
    }
    let text = match read_text(&full, &display) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return format!("{provenance}{display}: empty file");
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
    let mut out = provenance;
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

/// `read_docs` (RS-4): how another crate documents itself, at a version that
/// is stated in the answer. Local first, always; the network only where there
/// is no local copy and the user has opened `allow_online_docs`.
async fn tool_read_docs(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    online: bool,
) -> String {
    let Some(name) = arg_str(args, "crate").map(|s| s.trim()) else {
        return err("read_docs needs a string \"crate\", as in {\"crate\": \"serde\"}");
    };
    let version = arg_str(args, "version")
        .map(str::trim)
        .filter(|v| !v.is_empty());
    let path = arg_str(args, "path")
        .map(str::trim)
        .filter(|p| !p.is_empty());
    match crate::crate_docs::find_local(root, name, version) {
        crate::crate_docs::Local::Found(copy) => local_docs(&copy, path),
        crate::crate_docs::Local::Absent(why) => {
            if !online {
                return err(format!("{why}. {OFFLINE_HINT}"));
            }
            online_docs(name, version, path, &why).await
        }
    }
}

/// What to tell a model that asked for documentation this machine cannot reach
/// and is not permitted to fetch: the two ways to get it, both of them real.
const OFFLINE_HINT: &str = "read_docs reads only what cargo has already unpacked unless the \
                            user turns on allow_online_docs (`xencode config set \
                            allow_online_docs true`); a version named in Cargo.lock can also be \
                            unpacked on this machine with `cargo fetch`.";

/// The local half of `read_docs`: cargo's own unpacked copy, which is the
/// version this project builds unless the call asked for another one by name.
fn local_docs(copy: &crate::crate_docs::LocalCopy, path: Option<&str>) -> String {
    let dir = &copy.source.dir;
    let (rel, full) = match path {
        Some(requested) => {
            if !crate::crate_sources::path_is_contained(requested) {
                return err(format!(
                    "read_docs path {requested:?} reaches outside the crate; keep it relative \
                     and inside it"
                ));
            }
            let full = dir.join(requested);
            if !full.is_file() {
                return missing_doc(copy, &format!("{requested:?}"));
            }
            (requested.to_string(), full)
        }
        None => match crate::crate_docs::pick_doc_file(dir) {
            Some(found) => (found.rel, found.full),
            None => return missing_doc(copy, "readme"),
        },
    };
    let display = copy.source.spec_for(&rel);
    let text = match read_text(&full, &display) {
        Ok(text) => text,
        Err(e) => return e,
    };
    let body = crate::crate_docs::cap_doc(
        &text,
        &format!("read_file with path={display:?} and an offset pages through the rest"),
    );
    let others = crate::crate_docs::other_doc_files(dir, &rel);
    let mut out = format!(
        "[{} — read from {display}, unpacked by cargo]\n{body}",
        copy.label()
    );
    if !others.is_empty() {
        out.push_str(&format!(
            "\nOther documentation in this crate: {} — ask again with one of those paths.\n",
            others.join(", ")
        ));
    }
    out
}

/// The answer when the file a crate documents itself with is not there: name
/// the files that are, rather than send the model back to a search.
fn missing_doc(copy: &crate::crate_docs::LocalCopy, asked: &str) -> String {
    let others = crate::crate_docs::other_doc_files(&copy.source.dir, "");
    let listing = if others.is_empty() {
        format!(
            "nothing in {} reads like documentation; list_dir with path={:?} shows what is there",
            copy.source.name,
            copy.source.spec_for("")
        )
    } else {
        format!(
            "documentation it does have: {} — ask again with one of those paths",
            others.join(", ")
        )
    };
    err(format!(
        "{} {} has no {asked}; {listing}",
        copy.source.name, copy.source.version
    ))
}

/// `lookup_advisory` (RS-5): what the security corpora downloaded onto this
/// machine say about a crate. Reads only — and only from disk. The corpus is
/// fetched by the explicit `xencode advisories sync`, never from here, so a
/// turn inside the agent loop cannot produce a network request under the
/// name of a safety check.
fn tool_lookup_advisory(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    use xencode_analysis_rs::advisories as adv;

    let Some(name) = arg_str(args, "crate").map(str::trim) else {
        return err("lookup_advisory needs a string \"crate\", as in {\"crate\": \"chrono\"}");
    };
    let cache_dir = match xencode_config_rs::paths::cache_dir() {
        Ok(dir) => dir,
        Err(e) => {
            return err(format!(
                "lookup_advisory cannot locate the cache directory: {e}"
            ))
        }
    };
    let corpus = adv::corpus_dir(&cache_dir);
    let asked = arg_str(args, "version")
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    let pinned = if asked.is_none() {
        pinned_version(root, name)
    } else {
        None
    };
    let version = asked.or_else(|| pinned.clone());
    let lookup = match adv::advisories_for(&corpus, name) {
        Ok(lookup) => lookup,
        Err(e) => return err(e.to_string()),
    };
    let mut out = String::new();
    if let Some(pinned) = &pinned {
        out.push_str(&format!(
            "judging version {pinned}, which this project's Cargo.lock pins\n"
        ));
    }
    out.push_str(&adv::render_lookup(&lookup, version.as_deref()));
    out
}

/// The version of `crate_name` this workspace builds, from its lock file, when
/// the model asked about a crate without naming a version — so the answer is
/// about the dependency in front of it rather than a list to interpret.
fn pinned_version(root: &Path, crate_name: &str) -> Option<String> {
    crate::crate_sources::locked_packages_for(root)
        .into_iter()
        .find(|(name, _)| name == crate_name)
        .map(|(_, version)| version)
}

/// Read one installed skill's full instructions (M-3).
///
/// The name is looked up in the session's loaded skills and nothing else, which
/// is why this tool takes no path: a skill is reached by what the menu calls it
/// and no file outside the two scanned directories can be asked for. A name
/// that is not installed answers with the names that are, so the model can pick
/// one instead of guessing again.
fn tool_load_skill(
    skills: Option<&xencode_plugin_rs::SkillRuntime>,
    args: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let Some(name) = arg_str(args, "name").map(str::trim) else {
        return err("load_skill needs a string \"name\", copied from the Available skills list");
    };
    let Some(runtime) = skills else {
        return err(
            "load_skill is only available in the chat loop, where the session's skills are",
        );
    };
    if runtime.is_empty() {
        return err("no skills are installed, so there is nothing to load");
    }
    match runtime.get(name) {
        Some(skill) => runtime.render(skill),
        None => format!(
            "error: no skill named {name:?}. Installed: {}",
            runtime
                .skills()
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The fetched half: the two endpoints that publish by version, each reached
/// only because the copy on this machine was not readable.
async fn online_docs(name: &str, version: Option<&str>, path: Option<&str>, why: &str) -> String {
    // crates.io answers a version-less readme request with HTTP 400, so there
    // is no "latest" to fall back to; the version has to be in the call.
    let Some(version) = version else {
        return err(format!(
            "{why}, and a fetch needs a version to fetch: neither crates.io nor docs.rs will \
             answer for {name} without one, so pass version=\"…\" — the version you want, which \
             is not necessarily one this project builds."
        ));
    };
    let url = match path {
        Some(path) => crate::crate_docs::source_url(name, version, path),
        None => crate::crate_docs::readme_url(name, version),
    };
    let Ok(url) = url else {
        return err(url.unwrap_err());
    };
    let fetched = match crate::crate_docs::fetch(&url).await {
        Ok(fetched) => fetched,
        Err(e) => {
            return err(format!(
                "{e} — and the local copy was not readable either: {why}"
            ));
        }
    };
    let (status, body) = fetched;
    let what = path.unwrap_or("readme");
    if status == 404 {
        return err(format!(
            "{url} does not exist: {name} {version} is not published, or has no {what} in it."
        ));
    }
    if status != 200 {
        // A version with no rendered readme still redirects; the object store
        // it lands on is what refuses. That means "none published", not a
        // network failure, and the difference matters to what to try next.
        return err(format!(
            "{url} answered HTTP {status} after following crates.io's redirect, which is how the \
             service says {name} {version} has no {what} published."
        ));
    }
    let text = match path {
        Some(path) => match crate::crate_docs::docs_rs_source(&body) {
            Some(text) => text,
            None => {
                return err(format!(
                    "{url} is a page with no file on it — {name} {version} has no {path}."
                ))
            }
        },
        None => crate::crate_docs::html_to_text(&body),
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        return err(format!("{url} carries no text once its markup is removed."));
    }
    let body = crate::crate_docs::cap_doc(
        &text,
        &format!("the whole of it is at {url}, and `cargo fetch` puts a copy on this machine"),
    );
    format!("[{name} {version} {what} — fetched from {url}, because {why}]\n{body}")
}

/// The agent's one outbound read (RS-1): a URL the model named, fetched through
/// the guarded path and handed back as text.
///
/// `fetch_url_guarded`, never `fetch_url`. The difference is one argument at the
/// call site and nothing else — the address was chosen by a model that was
/// handed addresses by the pages, issues and files it reads, and the unguarded
/// path will fetch `http://10.0.0.8/` if asked, which makes this tool an
/// internal-network probe. Where the request landed is said out loud, in the
/// header line, so a page that redirected somewhere else reads as one.
async fn tool_web_fetch(args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(url) = arg_str(args, "url").filter(|u| !u.trim().is_empty()) else {
        return err("web_fetch needs a non-empty string \"url\"");
    };
    // The cap is the caller's to lower and not to raise: it exists to keep a
    // whole page out of the context window, so a bigger number buys nothing.
    let cap = args
        .get("max_chars")
        .and_then(|v| v.as_u64())
        .map(|v| (v as usize).clamp(1, xencode_analysis_rs::DEFAULT_TEXT_CAP_CHARS))
        .unwrap_or(xencode_analysis_rs::DEFAULT_TEXT_CAP_CHARS);
    let page =
        match xencode_analysis_rs::web::fetch_url_guarded(url).await {
            Ok(page) => page,
            // RS-7: the one branch worth the extra request. A 404 is the answer a
            // model most often gets when it guessed a documentation path, and a site
            // that publishes `llms.txt` has written down its real ones. Probed only
            // on a miss, never on a hit, because the file is absent across most of
            // the Rust ecosystem and a second request to every page to find nothing
            // is a tax, not a feature.
            Err(xencode_analysis_rs::FetchError::Status(404)) => {
                match llms_txt_fallback(url, cap).await {
                    Some(answer) => return answer,
                    // Nothing published, so say the page is missing and nothing
                    // more: not that an index exists elsewhere, and not as a
                    // refusal the caller could read as a permission problem.
                    None => return err(
                        "server returned status 404, and this site publishes no llms.txt index \
                         either — ask for an address you have actually seen, not a path you \
                         guessed"
                            .to_string(),
                    ),
                }
            }
            Err(e) => return err(e.to_string()),
        };
    let (text, dropped) = xencode_analysis_rs::cap_chars(&page.text, cap);
    let title = page.title.as_deref().unwrap_or("(no title)");
    let mut out = format!(
        "[{} — {title} — {} bytes fetched]\n{text}",
        page.url, page.bytes
    );
    if dropped > 0 {
        out.push_str(&format!(
            "\n[{dropped} more characters on this page; the whole text is at {}]",
            page.url
        ));
    }
    out
}

/// The site's own index for models, reached only after the page it was asked for
/// turned out to be missing. `None` means there is no `llms.txt` here — the
/// common case — which the caller reports as the 404 that it was.
///
/// The fallback goes to the same scheme, host and port the caller was approved
/// for, and through the same guard, so it is a second path on an address already
/// agreed to rather than a new trip.
async fn llms_txt_fallback(url: &str, cap: usize) -> Option<String> {
    let index = xencode_analysis_rs::web::llms_txt_url(url)?;
    let page = xencode_analysis_rs::web::fetch_url_guarded(&index)
        .await
        .ok()?;
    // An index is a table of contents, and a table of contents read as if it
    // were the page that was asked for is how a model ends up quoting an
    // index entry as documentation. The header says which of the two this is.
    let (text, dropped) = xencode_analysis_rs::cap_chars(&page.text, cap);
    let mut out = format!(
        "[{url} — 404 not found. What follows is this site's own index for models, at {}, \
         which is a list of its pages, not the page that was asked for]\n{text}",
        page.url
    );
    if dropped > 0 {
        out.push_str(&format!(
            "\n[{dropped} more characters in this index; the whole list is at {}]",
            page.url
        ));
    }
    Some(out)
}

/// How much of a result's snippet is handed to the model. Engines differ wildly
/// here — Tavily returns a paragraph it wrote, Wikipedia a sentence — and the
/// list is the thing being paid for, not the prose under it.
const SEARCH_SNIPPET_CHARS: usize = 240;

/// Results a call returns when the model asks for no number. Five is enough to
/// pick an address from and small enough that the list is not a page of its own.
const SEARCH_DEFAULT_RESULTS: usize = 5;

/// The agent's search call (RS-2): the question goes to the provider the person
/// named and the answer is a list of addresses with what the engine said about
/// each. No page is read here.
///
/// Keeping those two apart is the point of the shape. A list the model may then
/// follow one entry at a time, each with its own approval, is a different thing
/// from a tool that browses for it, which is why every provider caps the list and
/// why the snippet is truncated rather than handed over whole.
async fn tool_web_search(
    provider: &xencode_analysis_rs::SearchProvider,
    args: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let Some(query) = arg_str(args, "query").filter(|q| !q.trim().is_empty()) else {
        return err("web_search needs a non-empty string \"query\"");
    };
    // The cap is the caller's to lower and not to raise, exactly as a fetch's is:
    // it exists to keep a hundred links out of the context window.
    let want = args
        .get("max_results")
        .and_then(|v| v.as_u64())
        .map(|v| (v as usize).clamp(1, xencode_analysis_rs::MAX_SEARCH_RESULTS))
        .unwrap_or(SEARCH_DEFAULT_RESULTS);
    let hits = match xencode_analysis_rs::search_web(provider, query, want).await {
        Ok(hits) => hits,
        Err(e) => return err(e.to_string()),
    };
    let slug = provider.slug();
    if hits.is_empty() {
        // The engine answered. Saying "no results" as an error would invite a
        // retry of the same question, so this is stated as the flat answer it is.
        return format!(
            "[search — {slug} answered and found nothing for {query:?} — the wording is what \
             to change, not the setting]"
        );
    }
    let mut out = format!(
        "[search — {slug} — {} result(s) for {query:?}]\n",
        hits.len()
    );
    for (n, hit) in hits.iter().enumerate() {
        out.push_str(&format!(
            "{}. {}\n   {}\n",
            n + 1,
            truncate_one_line(&hit.title, 120),
            hit.url
        ));
        if !hit.snippet.is_empty() {
            out.push_str(&format!(
                "   {}\n",
                truncate_one_line(&hit.snippet, SEARCH_SNIPPET_CHARS)
            ));
        }
    }
    out.push_str(
        "\n[these are the addresses the engine named, not pages that have been read — reading \
         one is a separate request and a separate approval]",
    );
    out
}

/// The search provider the config names, resolved once when a run is built
/// (RS-2).
///
/// The credential is chosen from the provider's *name* and never tried in turn
/// across the two that take one: a Brave key must not reach Tavily's endpoint,
/// which is the same rule that keeps one model provider's key off another's
/// host. A setting that is half-filled comes back as `Err` carrying the words
/// that say which half, so the person hears "set the instance address" rather
/// than discovering the tool was never offered.
pub fn search_provider_from_config(
    config: &xencode_config_rs::XencodeConfig,
) -> Result<xencode_analysis_rs::SearchProvider, String> {
    let named = config.search_provider.trim();
    if named.is_empty() || named.eq_ignore_ascii_case("none") {
        return Ok(xencode_analysis_rs::SearchProvider::None);
    }
    let secret = match named.to_ascii_lowercase().as_str() {
        "brave" => Some(xencode_config_rs::SecretProvider::Brave),
        "tavily" => Some(xencode_config_rs::SecretProvider::Tavily),
        _ => None,
    };
    let key = match secret {
        Some(provider) => match config.api_keys.secret(provider) {
            Ok(key) => key,
            // A key that is present but unreadable is not an absent one. These are
            // different fixes — re-entering a value versus putting `secret-tool`
            // back on `PATH` — and collapsing them sends the person down the wrong
            // one.
            Err(problem) => {
                return Err(format!(
                    "web_search is set to `{named}` and its credential could not be read: {problem}"
                ))
            }
        },
        None => None,
    };
    xencode_analysis_rs::SearchProvider::parse(named, &config.search_searxng_url, key)
        .map_err(|e| e.to_string())
}

fn tool_list_dir(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let raw = arg_str(args, "path").filter(|s| !s.trim().is_empty());
    let (full, display, source) = match raw {
        Some(raw) => match readable_path(root, "list_dir", raw) {
            Ok(ok) => ok,
            Err(e) => return e,
        },
        None => (workspace_root(root), ".".to_string(), None),
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
    let provenance = crate_provenance(source.as_ref(), &full);
    if names.is_empty() {
        return format!("{provenance}{display}: empty");
    }
    if let Some(source) = source.as_ref() {
        names.push(format!(
            "… every path here is addressable as crate:{}/<path>",
            source.name
        ));
    }
    format!("{provenance}{display}:\n{}", names.join("\n"))
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
    let (scope, source) = match raw {
        Some(raw) => match readable_path(root, "search_files", raw) {
            Ok((full, _, source)) => (full, source),
            Err(e) => return e,
        },
        None => (workspace_root(root), None),
    };
    // Inside a locked crate the label each hit carries is the `crate:` address,
    // so a hit can be opened again without copying a registry path out of it.
    let display_of = |file: &Path| -> String {
        match source.as_ref().and_then(|s| s.inside(file)) {
            Some(inside) => source
                .as_ref()
                .map(|s| s.spec_for(&inside))
                .unwrap_or_default(),
            None => relative_display(root, file),
        }
    };
    let mut hits: Vec<String> = Vec::new();
    if scope.is_file() {
        let display = display_of(&scope);
        let _ = search_one_file(&scope, &display, &re, &mut hits);
    } else {
        let mut files = Vec::new();
        walk_files(&scope, 0, &mut files);
        files.sort();
        for file in files {
            let display = display_of(&file);
            let _ = search_one_file(&file, &display, &re, &mut hits);
            if hits.len() >= SEARCH_MAX_HITS {
                break;
            }
        }
    }
    let provenance = crate_provenance(source.as_ref(), &scope);
    if hits.is_empty() {
        return format!("{provenance}no matches for /{pattern}/");
    }
    let mut out = format!("{provenance}{} match(es):\n", hits.len());
    out.push_str(&hits.join("\n"));
    if hits.len() >= SEARCH_MAX_HITS {
        out.push_str(&format!(
            "\n… {SEARCH_MAX_HITS}-hit cap reached — narrow the pattern or path"
        ));
    }
    out
}

/// SE-5: a write or edit whose content carries a credential-shaped value gets
/// the summary the model reads back scrubbed, plus a line saying so. The bytes
/// on disk are what the user asked for and stay as written — this only guards
/// the *transcript copy* (the tool result, which rides into the model history,
/// the trace tail and the session recording). A fixture path or an allowlisted
/// tree is left alone, so a documented example key does not raise an alarm.
/// The patterns are the product's one credential list, shared with the trace
/// scrubber and the SE-4 taint gate.
fn secret_guard(rel: &str, content: &str, summary: String) -> String {
    if xencode_context_rs::path_skips_secret_scan(rel) {
        return summary;
    }
    if !xencode_context_rs::contains_secret(content) {
        return summary;
    }
    let scrubbed = xencode_context_rs::redact_secrets(&summary);
    format!(
        "[secret] the content written to {rel} carries a credential-shaped value. The \
         file on disk keeps the bytes as you wrote them; this summary has them redacted. \
         Revoke it and keep it out of the file — run /security-scan, or use a secrets \
         manager.\n{scrubbed}"
    )
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
    secret_guard(
        &display,
        content,
        format!(
            "{} {display} ({} line(s))\n{body}",
            if existed { "updated" } else { "created" },
            content.lines().count()
        ),
    )
}

/// EV-6: one note onto the scratchpad. There is no path argument on purpose —
/// the only file this can touch is `.xencode/notes.md`, and the cap, the
/// data-banner refusal and the redaction all happen in `xencode-context-rs` so
/// the pad reads the same however a line got into it.
fn tool_write_note(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(note) = arg_str(args, "note") else {
        return err("write_note needs a string \"note\"");
    };
    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
    let wrote = match xencode_context_rs::append_note(&xencode, note) {
        Ok(wrote) => wrote,
        Err(problem) => return err(format!("cannot write the note: {problem}")),
    };
    let kept = xencode_context_rs::note_lines(
        &std::fs::read_to_string(xencode_context_rs::notes_path(&xencode)).unwrap_or_default(),
    )
    .len();
    if wrote.added.is_empty() {
        let why = if wrote.stripped_data_lines > 0 {
            "it quoted fetched or tool output, which the pad does not keep as its own words"
        } else if wrote.duplicate_notes > 0 {
            "that note is already on the pad"
        } else {
            "there was nothing to write"
        };
        return format!("nothing written: {why}. {kept} note(s) on the pad.");
    }
    let mut out = format!(
        "noted: {} line(s); {} note(s) on the pad in .xencode/notes.md",
        wrote.added.len(),
        kept
    );
    if wrote.duplicate_notes > 0 {
        out.push_str(&format!(
            "; {} already there and left as it was",
            wrote.duplicate_notes
        ));
    }
    if wrote.stripped_data_lines > 0 {
        out.push_str(&format!(
            "; {} refused for quoting fetched or tool output",
            wrote.stripped_data_lines
        ));
    }
    if wrote.secrets_redacted > 0 {
        out.push_str(&format!(
            "; {} credential(s) taken out of the stored text",
            wrote.secrets_redacted
        ));
    }
    if !wrote.evicted.is_empty() {
        // The cap is the trap this tool ships with, so the model is told what it
        // just lost rather than finding a note missing three turns later.
        out.push_str(&format!(
            "; the pad holds {} notes and the oldest {} left it: {}. `/ctx fold` is how \
             a note becomes durable.",
            xencode_context_rs::NOTES_MAX_LINES,
            wrote.evicted.len(),
            wrote.evicted.join(" · ")
        ));
    }
    out
}

/// Result string fed back to the model when the user denies at the prompt.
/// Wording matters: weak models otherwise retry the identical call forever.
pub const DENIED_RESULT: &str = "error: the user denied this action. Do not retry it unchanged — explain, adjust, or ask the user.";

/// Result string fed back when a call needs approval and no person is there to
/// give it (an evaluation run, a headless task). Not "the user denied": told
/// that, a small model asks the absent user for permission and stops (SM-2,
/// measured 2026-10-08). This says what to do instead.
pub const UNATTENDED_RESULT: &str = "error: this call needs approval and nobody is here to give it, so it was not run. Do not ask for permission and do not retry it. Carry on with what needs no approval — read the files and edit them in the workspace — and finish the task.";

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
    if result == DENIED_RESULT || result == UNATTENDED_RESULT {
        CallOutcome::Denied
    } else if result == FORBIDDEN_RESULT {
        CallOutcome::Refused
    } else if result.starts_with("error:") {
        CallOutcome::Failed
    } else {
        CallOutcome::Finished
    }
}

/// The source line every tool result carries into the model's context (SE-2).
/// The system prompt declares tool results to be data fetched from the
/// machine — this line is what makes that claim checkable per result instead
/// of blanket. A server tool says `mcp`; a built-in tool says its own name
/// plus the argument it pointed at, so a `git log` result names the command.
///
/// The leading token is not typed out here: it comes from
/// [`xencode_context_rs::SourceClass`] (QK-3), the one vocabulary that says
/// whose bytes are whose. What class the result lands in is decided by the
/// tool's name — a fetch or a search is `Web`, an `mcp__…` tool is the server's
/// output, anything else ran on this machine.
pub fn mark_untrusted(call: &ToolCall, result: String) -> String {
    let class = xencode_context_rs::SourceClass::of_tool(&call.name);
    let token = class.marker().expect("every data class ships a marker");
    let label = if crate::mcp::is_mcp_tool(&call.name) {
        format!("mcp {}", call.name)
    } else {
        let args = call.arguments_object();
        let target = ["command", "path", "pattern", "query", "crate", "url"]
            .into_iter()
            .find_map(|key| args.get(key).and_then(|v| v.as_str()))
            .unwrap_or("");
        let target = truncate_one_line(target, 60);
        if target.is_empty() {
            call.name.clone()
        } else {
            format!("{} {target}", call.name)
        }
    };
    format!("{token}{label}\n{result}")
}

// ── Post-edit project checks (L-7) ────────────────────────────────────
// The model saying it is done is not the gate; the project's own commands
// exiting 0 is. These helpers decide which commands that is for a given
// workspace, and what a `run_command` result actually says about one.

/// The project's own test and lint commands, discovered from what is on disk:
/// a `Cargo.toml` in the workspace root means cargo, and cargo's own test and
/// clippy commands need no configuration to be true. Anything else returns
/// nothing — there is no invented command for a project we cannot read.
pub fn discover_check_commands(root: &std::path::Path) -> Vec<String> {
    if root.join("Cargo.toml").is_file() {
        vec!["cargo test".to_string(), "cargo clippy".to_string()]
    } else {
        Vec::new()
    }
}

/// What a check run's result string says, read back from the exact shapes
/// [`run_foreground`] produces. Only a real exit code counts: a denial, a
/// refusal, a timeout, or a missing toolchain all mean the edit was *not*
/// verified, which is different from verified-clean and must never be
/// reported as either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckVerdict {
    Passed,
    Failed,
    Unverifiable,
}

pub fn check_verdict(result: &str) -> CheckVerdict {
    // `error: ...` covers the denial/refusal constants, an unspawnable
    // command, and the timeout notice — none of which carry an exit code.
    if result.starts_with("error:") {
        return CheckVerdict::Unverifiable;
    }
    // The foreground runner always answers `$ <command>` then the status on
    // the second line; anything else is not a result it produced.
    match result.lines().nth(1) {
        Some("killed by signal") => CheckVerdict::Failed,
        Some(status) => match status.strip_prefix("exit ") {
            Some(code) => match code.trim().parse::<i32>() {
                Ok(0) => CheckVerdict::Passed,
                Ok(_) => CheckVerdict::Failed,
                Err(_) => CheckVerdict::Unverifiable,
            },
            None => CheckVerdict::Unverifiable,
        },
        None => CheckVerdict::Unverifiable,
    }
}

/// One-line label for the approval overlay: tool + its focus argument.
pub fn approval_summary(call: &ToolCall) -> String {
    let args = call.arguments_object();
    let focus = arg_str(&args, "path")
        .or_else(|| arg_str(&args, "command"))
        .or_else(|| arg_str(&args, "pattern"))
        // The address is the whole decision for a fetch, so it is the line the
        // overlay leads with. A summary of `web_fetch` without it would ask
        // about a trip without saying where.
        .or_else(|| arg_str(&args, "url"))
        // Same for a search (RS-2): the question is what leaves, and it is the
        // only thing the person can judge the trip on.
        .or_else(|| arg_str(&args, "query"))
        .unwrap_or("");
    if focus.is_empty() {
        call.name.clone()
    } else {
        truncate_one_line(&format!("{} {}", call.name, focus), 90)
    }
}

/// What the overlay shows for one pending call: the body text, and the files
/// whose bytes that text was worked out from.
pub struct ApprovalShown {
    pub preview: String,
    pub draft: ApprovalDraft,
}

/// The shown diff and the bound bytes, taken in one pass over the tree. They
/// have to come from the same pass: a preview read at one moment and a
/// fingerprint read at another can describe two different changes, and then the
/// re-check would be comparing the approval against bytes nobody was shown.
pub fn approval_shown(root: &Path, call: &ToolCall) -> ApprovalShown {
    let (preview, bound) = preview_and_bound_files(root, call);
    ApprovalShown {
        preview,
        draft: ApprovalDraft::bind(&bound),
    }
}

/// The overlay's body: the concrete bytes at stake. File writes/edits get
/// the exact unified diff of the proposed change (worked out from the file as
/// it stands when the prompt is raised — [`approval_draft`] binds the approval
/// to those bytes, so a file edited mid-prompt is re-shown rather than
/// overwritten); shell tools get the
/// literal command line; everything else gets the argument summary.
pub fn approval_preview(root: &Path, call: &ToolCall) -> String {
    approval_shown(root, call).preview
}

/// The preview text plus, for the calls whose text is a diff of file contents,
/// exactly which files it is a diff of.
fn preview_and_bound_files(root: &Path, call: &ToolCall) -> (String, Vec<(PathBuf, String)>) {
    let args = call.arguments_object();
    let diff_for = |path: &str, new_text: &str| -> (String, Vec<(PathBuf, String)>) {
        let (full, display) = match workspace_path(root, path) {
            Ok(ok) => ok,
            Err(e) => return (e, Vec::new()),
        };
        let exists = full.exists();
        let old = if exists {
            match read_text(&full, &display) {
                Ok(t) => t,
                Err(e) => return (e, Vec::new()),
            }
        } else {
            String::new()
        };
        let header = if exists {
            format!("target: {display}")
        } else {
            format!("target: {display} (new file)")
        };
        (
            format!("{header}\n{}", unified_diff(&old, new_text).trim_end()),
            vec![(full, display)],
        )
    };
    let files_of = |rewrites: &[PlannedRewrite]| -> Vec<(PathBuf, String)> {
        rewrites
            .iter()
            .map(|r| (r.full.clone(), r.display.clone()))
            .collect()
    };
    match call.name.as_str() {
        "write_file" => match (arg_str(&args, "path"), arg_str(&args, "content")) {
            (Some(p), Some(c)) => diff_for(p, c),
            _ => (summarize_call(call), Vec::new()),
        },
        "edit_file" => {
            let (Some(p), Some(old), Some(new)) = (
                arg_str(&args, "path"),
                arg_str(&args, "old"),
                arg_str(&args, "new"),
            ) else {
                return (summarize_call(call), Vec::new());
            };
            if old.is_empty() {
                return (summarize_call(call), Vec::new());
            }
            let (full, _) = match workspace_path(root, p) {
                Ok(ok) => ok,
                Err(e) => return (e, Vec::new()),
            };
            let Ok(current) = read_text(&full, p) else {
                return (summarize_call(call), Vec::new());
            };
            // The same computation the tool runs, so the diff approved here is the
            // diff that gets written.
            let (updated, note) = match plan_text_edit(&current, old, new, arg_bool(&args, "all")) {
                PlannedEdit::Exact { updated, .. } => (updated, None),
                PlannedEdit::Loose { updated } => (
                    updated,
                    Some(
                        "\n(note: \"old\" matched one block only once each line's surrounding \
                          whitespace was ignored; the new text takes the file's indentation)"
                            .to_string(),
                    ),
                ),
                PlannedEdit::NotFound => (
                    current.clone(),
                    Some(
                        "\n(note: \"old\" is not in the file, even ignoring whitespace — the \
                          edit would fail)"
                            .to_string(),
                    ),
                ),
                PlannedEdit::Ambiguous { count } => (
                    current.clone(),
                    Some(format!(
                        "\n(note: \"old\" currently matches {count} times — the edit \
                         would fail unless all=true)"
                    )),
                ),
            };
            let (mut preview, bound) = diff_for(p, &updated);
            if let Some(note) = note {
                preview.push_str(&note);
            }
            (preview, bound)
        }
        "edit_symbol" => match planned_symbol_edit(root, &args) {
            Err(reason) => (reason, Vec::new()),
            Ok((full, display, current, updated)) => (
                format!(
                    "target: {display}\n{}",
                    unified_diff(&current, &updated).trim_end()
                ),
                vec![(full, display)],
            ),
        },
        "ast_edit" => match plan_ast_edit(root, &args, AST_PREVIEW_TIMEOUT_SECS) {
            // The same planner the executor uses, so the preview cannot describe
            // a different edit from the one that lands.
            Err(reason) => (reason, Vec::new()),
            Ok(plan) if plan.replacement.is_none() => (
                format!(
                    "ast_edit would report {} site(s) and change nothing:\n{}",
                    plan.sites.len(),
                    plan.sites.join("\n")
                ),
                Vec::new(),
            ),
            Ok(plan) => {
                let bound = files_of(&plan.rewrites);
                let mut out = format!(
                    "ast_edit would rewrite {} site(s) across {} file(s):",
                    plan.rewrites.iter().map(|r| r.sites).sum::<usize>(),
                    plan.rewrites.len()
                );
                out.push_str(&describe_rewrites(&plan.rewrites));
                (out, bound)
            }
        },
        "rename" => match plan_rename(root, &args, AST_PREVIEW_TIMEOUT_SECS) {
            Err(reason) => (reason, Vec::new()),
            Ok(plan) => {
                let bound = files_of(&plan.rewrites);
                let total: usize = plan.rewrites.iter().map(|r| r.sites).sum();
                (
                    format!(
                        "rename would turn `{}` ({} in {}) into `{}` at {total} site(s) across {} file(s):{}\n\n{}",
                        plan.symbol,
                        plan.kind,
                        plan.definition,
                        plan.new_name,
                        plan.rewrites.len(),
                        describe_rewrites(&plan.rewrites),
                        plan.basis
                    ),
                    bound,
                )
            }
        },
        "codemod" => match plan_codemod(root, &args, AST_PREVIEW_TIMEOUT_SECS) {
            Err(reason) => (reason, Vec::new()),
            Ok(plan) if !plan.rewrites_code => (
                format!(
                    "codemod would report {} site(s) and change nothing:\n{}",
                    plan.sites.len(),
                    plan.sites.join("\n")
                ),
                Vec::new(),
            ),
            Ok(plan) => {
                let bound = files_of(&plan.rewrites);
                let mut out = format!(
                    "codemod would rewrite {} site(s) across {} file(s):",
                    plan.rewrites.iter().map(|r| r.sites).sum::<usize>(),
                    plan.rewrites.len()
                );
                out.push_str(&describe_rewrites(&plan.rewrites));
                if !plan.entangled.is_empty() {
                    out.push_str(&format!(
                        "\n(note: {} of those file(s) already have uncommitted changes: {})",
                        plan.entangled.len(),
                        plan.entangled.join(", ")
                    ));
                }
                (out, bound)
            }
        },
        "background_start" | "run_command" => match arg_str(&args, "command") {
            Some(command) if !command.trim().is_empty() => {
                (format!("command: sh -c {command:?}"), Vec::new())
            }
            _ => (summarize_call(call), Vec::new()),
        },
        "debug_test" => (
            format!(
                "debug: `cargo test --no-run`{}, then run test `{}` under a debugger \
                 until {}:{}, read its locals, and end the session",
                arg_str(&args, "package")
                    .map(|p| format!(" -p {p}"))
                    .unwrap_or_default(),
                arg_str(&args, "test").unwrap_or("?"),
                arg_str(&args, "file").unwrap_or("?"),
                args.get("line").and_then(|v| v.as_u64()).unwrap_or(0)
            ),
            Vec::new(),
        ),
        // The one call whose entire consequence is its argument, so the preview
        // can say more than what will happen — it can say whether it is allowed
        // to happen at all. Same check the fetch makes, one host resolution,
        // shown before the person answers rather than after.
        "web_fetch" => match arg_str(&args, "url") {
            Some(url) => {
                let mut out = format!("fetch: {url}");
                out.push_str(&match xencode_analysis_rs::web::guard_destination(url) {
                    // Not "a public address": loopback passes the guard on
                    // purpose, and a label that called `127.0.0.1` public would
                    // teach the reader to distrust the label. The honest pair is
                    // whether the request would go out or be refused. The
                    // second sentence names the one other trip this call can
                    // make, which is the same host and a different path.
                    Ok(()) => "  (would connect; the page is returned as text, capped; \
                               if it turns out to be missing, this site's own /llms.txt \
                               index is asked for on the same address)"
                        .to_string(),
                    Err(e) => format!("  — this one would be refused: {e}"),
                });
                (out, Vec::new())
            }
            _ => (summarize_call(call), Vec::new()),
        },
        // RS-2: where a fetch's whole consequence is one address, a search's is
        // the question and the engine it goes to. The engine is a config value
        // rather than a call argument, so the preview says where to read it
        // instead of pretending to know it, and says plainly that the answer is a
        // list of links — not those links read.
        "web_search" => match arg_str(&args, "query") {
            Some(query) if !query.trim().is_empty() => (
                format!(
                    "search: {}\n  (the question is sent to the search provider named in the \
                     config — `xencode config show` says which one that is — and what comes back \
                     is titles, addresses and short snippets. Nothing in that list is read.)",
                    truncate_one_line(query, 200)
                ),
                Vec::new(),
            ),
            _ => (summarize_call(call), Vec::new()),
        },
        _ => (summarize_call(call), Vec::new()),
    }
}

/// How many occurrences `occurrence_context` names before saying how many it
/// left out, and how many near-miss blocks `absence_candidates` shows.
const MATCH_REPORT_CAP: usize = 6;

/// 1-based line number of a byte offset in `text`.
fn line_at(text: &str, byte: usize) -> usize {
    text[..byte].matches('\n').count() + 1
}

/// A `L{a}-L{b}`-style window label: one number when the block is one line.
fn window_label(start: usize, end: usize) -> String {
    if start == end {
        format!("line {start}")
    } else {
        format!("lines {start}-{end}")
    }
}

/// One `old` occurrence with a line of context on each side, exactly as the
/// file holds it. The tab after the line number is what `read_file` shows, so
/// copying from either output behaves the same.
fn block_with_context(text: &str, byte: usize, needle: &str) -> String {
    let before = &text[..byte];
    let start_line = before.matches('\n').count() + 1;
    let needle_lines = needle.lines().count().max(1);
    let end_line = start_line + needle_lines - 1;
    let lines: Vec<&str> = text.lines().collect();
    let from = start_line.saturating_sub(2).max(1);
    let to = (end_line + 1).min(lines.len());
    let mut out = String::new();
    for n in from..=to {
        let mark = if n >= start_line && n <= end_line {
            "→"
        } else {
            " "
        };
        let content = truncate_one_line(lines[n - 1], 120);
        out.push_str(&format!("{mark} {n:>4} | {content}\n"));
    }
    out.trim_end().to_string()
}

/// Where every `old` actually is, so a non-unique edit self-corrects from the
/// report instead of guessing new context.
fn occurrence_context(text: &str, old: &str, count: usize) -> String {
    let mut out = String::new();
    for (i, byte) in text
        .match_indices(old)
        .map(|(p, _)| p)
        .take(MATCH_REPORT_CAP)
        .enumerate()
    {
        let start = line_at(text, byte);
        let end = start + old.lines().count().max(1) - 1;
        out.push_str(&format!(
            "\n  match {} of {count} ({}):\n{}",
            i + 1,
            window_label(start, end),
            block_with_context(text, byte, old)
        ));
    }
    if count > MATCH_REPORT_CAP {
        out.push_str(&format!("\n  … and {} more", count - MATCH_REPORT_CAP));
    }
    out
}

/// Why a zero-match `old` probably missed: the lines the file holds that the
/// model's text looks like — identical modulo whitespace first, then blocks
/// whose first line matches but whose rest drifted. Always labelled as
/// near misses: nothing here is proposed as a match, and nothing here is
/// written; the exact-match contract is unchanged.
fn absence_candidates(text: &str, old: &str) -> String {
    let fold = |s: &str| -> String { s.split_whitespace().collect::<Vec<_>>().join(" ") };
    let old_fold = fold(old);
    let lines: Vec<&str> = text.lines().collect();
    let mut hits: Vec<String> = Vec::new();
    // Same words, different spacing: the usual way an `old` misses.
    let mut start = 0usize;
    while start < lines.len() && hits.len() < MATCH_REPORT_CAP {
        let span = old.lines().count().max(1);
        if start + span <= lines.len() {
            let window = fold(&lines[start..start + span].join(" "));
            if window == old_fold {
                hits.push(format!(
                    "\n  {} — same text, different whitespace:\n{}",
                    window_label(start + 1, start + span),
                    block_with_context(
                        text,
                        offset_of_line(text, start),
                        &lines[start..start + span].join("\n")
                    )
                ));
                start += span;
                continue;
            }
        }
        start += 1;
    }
    // First line present, rest not: name how far it matched.
    if hits.is_empty() {
        let first = old.lines().next().unwrap_or("");
        let first_fold = fold(first);
        if !first_fold.is_empty() {
            for (i, line) in lines.iter().enumerate() {
                if hits.len() >= MATCH_REPORT_CAP {
                    break;
                }
                if fold(line) == first_fold {
                    let matched = old
                        .lines()
                        .zip(lines[i..].iter())
                        .take_while(|(a, b)| fold(a) == fold(b))
                        .count();
                    hits.push(format!(
                        "\n  line {} — {} of {} lines match (ignoring whitespace):\n{}",
                        i + 1,
                        matched,
                        old.lines().count(),
                        block_with_context(text, offset_of_line(text, i), first)
                    ));
                }
            }
        }
    }
    if hits.is_empty() {
        return " — nothing in the file resembles it".to_string();
    }
    hits.concat()
}

/// Byte offset where the (0-based) line index begins.
fn offset_of_line(text: &str, line_index: usize) -> usize {
    let mut off = 0usize;
    for _ in 0..line_index {
        match text[off..].find('\n') {
            Some(n) => off += n + 1,
            None => break,
        }
    }
    off
}

/// What an `edit_file` call would do to a file's text.
#[derive(Debug, PartialEq)]
enum PlannedEdit {
    /// `old` was found as written; `count` replacements were made.
    Exact { updated: String, count: usize },
    /// `old` was found only once each line's surrounding whitespace was ignored,
    /// at exactly one place; `new` was re-indented to match the file (SM-2).
    Loose { updated: String },
    /// Nowhere, even ignoring whitespace.
    NotFound,
    /// More than once as written, and `all` was not asked for.
    Ambiguous { count: usize },
}

/// The one computation behind both the `edit_file` tool and its approval
/// preview, so what a person approves is what gets written.
///
/// An exact match always wins. Only when there is none is the match retried line
/// by line with each line's leading and trailing whitespace ignored, and only a
/// single matching block is used: a small model reliably finds the right lines
/// but not their exact indentation — on 2026-10-08 Qwen3-4B sent the same
/// mis-indented 118 bytes twice after being told to copy them exactly (SM-1).
fn plan_text_edit(text: &str, old: &str, new: &str, all: bool) -> PlannedEdit {
    let count = text.matches(old).count();
    if count > 1 && !all {
        return PlannedEdit::Ambiguous { count };
    }
    if count >= 1 {
        let updated = if all {
            text.replace(old, new)
        } else {
            text.replacen(old, new, 1)
        };
        return PlannedEdit::Exact {
            updated,
            count: if all { count } else { 1 },
        };
    }
    let wanted: Vec<&str> = old.lines().map(str::trim).collect();
    let first = wanted.iter().position(|l| !l.is_empty());
    let last = wanted.iter().rposition(|l| !l.is_empty());
    let (Some(first), Some(last)) = (first, last) else {
        return PlannedEdit::NotFound;
    };
    let wanted = &wanted[first..=last];
    // Line starts and ends, so a match maps back to bytes in the original text.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for line in text.split_inclusive('\n') {
        let body = line.trim_end_matches(['\n', '\r']);
        spans.push((start, start + body.len()));
        start += line.len();
    }
    let matches: Vec<usize> = (0..spans.len().saturating_sub(wanted.len() - 1))
        .filter(|&i| {
            wanted
                .iter()
                .enumerate()
                .all(|(k, w)| text[spans[i + k].0..spans[i + k].1].trim() == *w)
        })
        .collect();
    let [at] = matches.as_slice() else {
        return if matches.is_empty() {
            PlannedEdit::NotFound
        } else {
            PlannedEdit::Ambiguous {
                count: matches.len(),
            }
        };
    };
    let (block_start, _) = spans[*at];
    let (_, block_end) = spans[*at + wanted.len() - 1];
    // Re-indent `new` from the indentation `old` claimed to the one the file has.
    let indent_of = |s: &str| s.len() - s.trim_start().len();
    let file_line = &text[block_start..spans[*at].1];
    let file_indent = &file_line[..indent_of(file_line)];
    let old_first = old.lines().nth(first).unwrap_or("");
    let old_indent = &old_first[..indent_of(old_first)];
    let reindented: Vec<String> = new
        .trim_matches('\n')
        .lines()
        .map(|line| match line.strip_prefix(old_indent) {
            Some(rest) if !line.trim().is_empty() => format!("{file_indent}{rest}"),
            _ => line.to_string(),
        })
        .collect();
    let line_end = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut updated = String::with_capacity(text.len() + new.len());
    updated.push_str(&text[..block_start]);
    updated.push_str(&reindented.join(line_end));
    updated.push_str(&text[block_end..]);
    PlannedEdit::Loose { updated }
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
    let (updated, how) = match plan_text_edit(&text, old, new, arg_bool(args, "all")) {
        PlannedEdit::Exact { updated, count } => {
            (updated, format!("replaced {count} occurrence(s)"))
        }
        PlannedEdit::Loose { updated } => (
            updated,
            "replaced 1 block that matched once each line's surrounding whitespace was \
             ignored; the new text was given the file's indentation"
                .to_string(),
        ),
        PlannedEdit::NotFound => {
            return err(format!(
                "\"old\" not found in {display} (0 matches, even ignoring each line's \
                 surrounding whitespace).{}\nRe-issue edit_file with lines taken from \
                 read_file {display}.",
                absence_candidates(&text, old)
            ))
        }
        PlannedEdit::Ambiguous { count } => {
            return err(format!(
                "\"old\" appears {count} times in {display} — every match below. Pass more \
                 surrounding context to make \"old\" unique, or all=true to replace every \
                 occurrence.\n{}",
                occurrence_context(&text, old, count)
            ))
        }
    };
    if let Err(e) = std::fs::write(&full, &updated) {
        return err(format!("cannot write {display}: {e}"));
    }
    let diff = unified_diff(&text, &updated);
    secret_guard(
        &display,
        &updated,
        format!("edited {display}: {how}\n{diff}")
            .trim_end()
            .to_string(),
    )
}

/// The edit an `edit_symbol` call would perform, or the reason it would refuse.
/// A refusal comes back as the finished error string, so a caller can hand it
/// straight to the model.
///
/// Shared by the tool and the approval overlay on purpose: what a person is shown
/// before approving has to be the same computation that produces the bytes, or the
/// approval is for one change and the write is another. Both read the file when
/// asked, so an edit made while a prompt is open is caught twice over: the gate
/// re-checks the bytes the preview was worked out from ([`approval_draft`]) before
/// spending the approval, and the tool works the change out again from what it
/// finds.
fn planned_symbol_edit(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
) -> Result<(PathBuf, String, String, String), String> {
    let Some(raw) = arg_str(args, "path") else {
        return Err(err("edit_symbol needs a string \"path\""));
    };
    let Some(symbol) = arg_str(args, "symbol") else {
        return Err(err("edit_symbol needs a string \"symbol\""));
    };
    let Some(new_body) = arg_str(args, "new_body") else {
        return Err(err("edit_symbol needs a string \"new_body\""));
    };
    let (full, display) = workspace_path(root, raw)?;
    // Asked before the parser: a valid file in another language is refused as the
    // language it is, not as code that fails to parse.
    if let Some(refusal) = xencode_context_rs::semantic_tier_refusal(
        "Symbol-level editing",
        &display,
        xencode_context_rs::detect_language(&full),
    ) {
        return Err(err(refusal));
    }
    let text = read_text(&full, &display)?;
    match xencode_context_rs::replace_symbol_body(&text, symbol, new_body) {
        Err(failure) => Err(err(failure.to_message())),
        Ok(updated) => Ok((full, display, text, updated)),
    }
}

fn tool_edit_symbol(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let (full, display, text, updated) = match planned_symbol_edit(root, args) {
        Ok(plan) => plan,
        Err(reason) => return reason,
    };
    if let Err(e) = std::fs::write(&full, &updated) {
        return err(format!("cannot write {display}: {e}"));
    }
    let diff = unified_diff(&text, &updated);
    format!(
        "edited {display}: replaced the body of {}\n{diff}",
        quote_symbol(args)
    )
    .trim_end()
    .to_string()
}

/// The symbol an `edit_symbol` call named, for the line that reports what changed.
fn quote_symbol(args: &serde_json::Map<String, serde_json::Value>) -> String {
    match arg_str(args, "symbol") {
        Some(symbol) => format!("`{symbol}`"),
        None => "a declaration".to_string(),
    }
}

/// One file an `ast_edit` call would rewrite, with the text it currently holds.
#[derive(Debug)]
pub(crate) struct PlannedRewrite {
    full: PathBuf,
    display: String,
    current: String,
    updated: String,
    sites: usize,
}

/// What an `ast_edit` call found, whether or not it is going to write.
#[derive(Debug)]
pub struct AstEditPlan {
    /// Human-readable line per match: `path:line: matched text`.
    sites: Vec<String>,
    /// Files to rewrite. Empty in search-only mode.
    rewrites: Vec<PlannedRewrite>,
    replacement: Option<String>,
}

/// The ast-grep executable, or the reason there isn't one.
///
/// A missing binary is reported, never treated as "no sites matched": those two
/// look identical from the outside and only one of them is a fact about the code.
fn ast_grep_binary() -> Result<String, String> {
    // The analysis crate checks that an `sg` on PATH really is ast-grep: on
    // Ubuntu it is usually shadow-utils' "run as another group".
    xencode_analysis_rs::runtime_hazards::ast_grep_binary()
        .map_err(|_| err(missing_ast_grep_message()))
}

/// What a caller is told when the pattern engine is not on this machine.
///
/// Split out so the wording is assertable without emptying `PATH` in a test:
/// these tests run in parallel, and a process-global environment variable is
/// not something one of them gets to own.
fn missing_ast_grep_message() -> String {
    "ast_edit needs the `ast-grep` binary and neither `ast-grep` nor `sg` is on PATH, \
     so the pattern was not run and nothing is known about the code. Install it with \
     `npm i -g @ast-grep/cli` or `cargo install ast-grep`, or use edit_symbol / \
     search_files, which need no external binary."
        .to_string()
}

/// One site ast-grep matched, as far as this tool needs it.
#[derive(Debug)]
struct AstMatch {
    file: String,
    line: usize,
    column: usize,
    text: String,
    /// Byte span of the match inside `file`. Absent when ast-grep reported no
    /// `range`, which makes the site searchable but not rewritable.
    span: Option<(usize, usize)>,
    /// What the match becomes, present only in a rewrite run.
    replacement: Option<String>,
}

/// Read `--json` output. The top level is a bare array of matches, and an empty
/// result is `[]` — including for a pattern ast-grep could not parse, which it
/// reports no differently from a pattern that simply found nothing.
fn parse_ast_matches(stdout: &str) -> Result<Vec<AstMatch>, String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| err(format!("ast-grep printed output that is not JSON: {e}")))?;
    let array = value
        .as_array()
        .ok_or_else(|| err("ast-grep printed JSON that is not a list of matches"))?;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let Some(obj) = item.as_object() else {
            return Err(err("ast-grep printed a match that is not an object"));
        };
        let file = obj
            .get("file")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let range = obj.get("range");
        let start = range.and_then(|r| r.get("start"));
        let line = start
            .and_then(|s| s.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as usize
            + 1;
        let column = start
            .and_then(|s| s.get("column"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize
            + 1;
        let offsets = range
            .and_then(|r| r.get("byteOffset"))
            .and_then(|b| b.as_object());
        let span = match (
            offsets
                .and_then(|b| b.get("start"))
                .and_then(|v| v.as_u64()),
            offsets.and_then(|b| b.get("end")).and_then(|v| v.as_u64()),
        ) {
            (Some(start), Some(end)) if end >= start => Some((start as usize, end as usize)),
            _ => None,
        };
        out.push(AstMatch {
            file,
            line,
            column,
            text: obj
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            span,
            replacement: obj
                .get("replacement")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        });
    }
    Ok(out)
}

/// Run ast-grep over `target` and return what it matched.
fn run_ast_grep(
    root: &Path,
    pattern: &str,
    target: &str,
    replacement: Option<&str>,
    language: Option<&str>,
    timeout_secs: u64,
) -> Result<Vec<AstMatch>, String> {
    let binary = ast_grep_binary()?;
    let mut command = std::process::Command::new(&binary);
    // The child runs in the workspace root, because `target` is relative to it.
    // Without this, a relative path resolves against xencode's own working
    // directory and the search silently finds nothing.
    command.current_dir(root);
    command
        .arg("run")
        .arg("--pattern")
        .arg(pattern)
        .arg("--json");
    if let Some(replacement) = replacement {
        command.arg("--rewrite").arg(replacement);
    }
    if let Some(language) = language {
        command.arg("--lang").arg(language);
    }
    command.arg(target);

    let output = run_with_timeout(&mut command, timeout_secs)
        .map_err(|e| err(format!("could not run {binary}: {e}")))?;

    // ast-grep exits 1 for "nothing matched" and for a bad pattern alike, and
    // prints `[]` for both, so the exit code carries no information here. Only
    // a crash is distinguishable, and only by whether stdout parsed.
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        if !stderr.is_empty() && !stderr.starts_with("ERROR: ") {
            return Err(err(format!("ast-grep failed: {stderr}")));
        }
    }
    parse_ast_matches(&String::from_utf8_lossy(&output.stdout))
}

/// Run a command, killing it after `timeout_secs`. Returns what it produced.
fn run_with_timeout(
    command: &mut std::process::Command,
    timeout_secs: u64,
) -> std::io::Result<std::process::Output> {
    use std::process::Stdio;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs.max(1));
    loop {
        match child.try_wait()? {
            Some(_) => return child.wait_with_output(),
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "ast-grep did not finish in time",
                ));
            }
            None => std::thread::sleep(std::time::Duration::from_millis(25)),
        }
    }
}

/// Work out what an `ast_edit` call would do, without touching a file.
///
/// Split from the write so the approval preview can show the real diff, and so a
/// rewrite is computed in memory and written once, atomically, per file.
pub fn plan_ast_edit(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> Result<AstEditPlan, String> {
    let Some(pattern) = arg_str(args, "pattern") else {
        return Err(err("ast_edit needs a string \"pattern\""));
    };
    if pattern.trim().is_empty() {
        return Err(err("ast_edit needs a non-empty \"pattern\""));
    }
    let Some(raw) = arg_str(args, "path") else {
        return Err(err("ast_edit needs a string \"path\""));
    };
    let replacement = arg_str(args, "replacement");
    let language = arg_str(args, "language");

    let (full, display) = workspace_path(root, raw)?;
    if !full.exists() {
        return Err(err(format!("{display} does not exist")));
    }
    // Asked before the parser, for the same reason `edit_symbol` asks: a valid
    // file in another language is refused as the language it is.
    if full.is_file() {
        if let Some(refusal) = xencode_context_rs::semantic_tier_refusal(
            "Structural search and rewrite",
            &display,
            xencode_context_rs::detect_language(&full),
        ) {
            return Err(err(refusal));
        }
    }

    // Handed the root-relative path, not the absolute one, so the paths ast-grep
    // reports — and therefore the paths in the transcript and the diff — read
    // `src/lib.rs` rather than whatever this machine happens to call it. The
    // child is run in `root` so that relative path means what it says.
    let matches = run_ast_grep(root, pattern, &display, replacement, language, timeout_secs)?;

    if matches.is_empty() {
        // The recorded trap: a pattern that matches nothing is indistinguishable
        // from a pattern that does not parse, and both look like success from
        // the outside. Say so rather than reporting a clean sweep.
        let mut message = format!(
            "ast_edit matched no sites in {display} for pattern `{pattern}`, and \
             nothing was changed. That result cannot distinguish a pattern that is \
             wrong from code that does not contain it"
        );
        if let Some(language) = &language {
            message.push_str(&format!(" (language hint `{language}`)"));
        }
        message.push_str(
            ". Check the metavariables are written $NAME, and that the shape is \
             present, before reading this as \"the code is already correct\".",
        );
        return Err(err(message));
    }

    let sites = describe_sites(&matches);

    let Some(replacement) = replacement else {
        return Ok(AstEditPlan {
            sites,
            rewrites: Vec::new(),
            replacement: None,
        });
    };

    let rewrites = plan_file_rewrites(root, &matches, ToolName::AstEdit)?;
    Ok(AstEditPlan {
        sites,
        rewrites,
        replacement: Some(replacement.to_string()),
    })
}

/// One `path:line` line per site, for the report and the preview.
fn describe_sites(matches: &[AstMatch]) -> Vec<String> {
    matches
        .iter()
        .map(|m| {
            let first_line = m.text.lines().next().unwrap_or("").trim_end();
            let shown = if m.text.contains('\n') {
                format!("{first_line} …")
            } else {
                first_line.to_string()
            };
            format!("{}:{}:{} {shown}", m.file, m.line, m.column)
        })
        .collect()
}

/// Which tool asked, for the wording of a refusal.
#[derive(Clone, Copy, PartialEq)]
enum ToolName {
    AstEdit,
    Codemod,
    Rename,
}

impl ToolName {
    fn label(self) -> &'static str {
        match self {
            ToolName::AstEdit => "ast_edit",
            ToolName::Codemod => "codemod",
            ToolName::Rename => "rename",
        }
    }
}

/// Group matches by file and compute each file's new text, in memory.
///
/// Shared by `ast_edit` and `codemod` so both derive their bytes the same way
/// and the approval preview cannot describe a different edit from the one that
/// lands. Edits within a file are spliced from the back, so an earlier match's
/// byte offsets stay valid while a later one is applied.
fn plan_file_rewrites(
    root: &Path,
    matches: &[AstMatch],
    tool: ToolName,
) -> Result<Vec<PlannedRewrite>, String> {
    let mut order: Vec<String> = Vec::new();
    for m in matches {
        if !order.contains(&m.file) {
            order.push(m.file.clone());
        }
    }
    let mut rewrites = Vec::new();
    for file in order {
        let sites_here: Vec<&AstMatch> = matches.iter().filter(|m| m.file == file).collect();
        let (file_full, file_display) = match workspace_path(root, &file) {
            Ok(ok) => ok,
            // ast-grep reports a path as it was given it, so a file found through
            // a directory argument comes back root-relative already. Anything
            // that does not resolve stays inside the root, or is dropped.
            Err(_) => continue,
        };
        let current = read_text(&file_full, &file_display)?;
        let mut spans: Vec<(usize, usize, &str)> = Vec::with_capacity(sites_here.len());
        for m in &sites_here {
            let (Some(start), Some(end)) = (m.span.map(|s| s.0), m.span.map(|s| s.1)) else {
                return Err(err(format!(
                    "{} found a match in {file_display} without a byte range, so it \
                     cannot be rewritten safely. Run it without a fix to see the sites.",
                    tool.label()
                )));
            };
            let Some(replacement) = &m.replacement else {
                return Err(err(format!(
                    "{} did not return rewritten text for the match at {file_display}:{}, \
                     so nothing was changed",
                    tool.label(),
                    m.line
                )));
            };
            spans.push((start, end, replacement));
        }
        let updated = splice_replacements(&current, &spans, &file_display)?;
        rewrites.push(PlannedRewrite {
            full: file_full,
            display: file_display,
            current,
            updated,
            sites: sites_here.len(),
        });
    }
    Ok(rewrites)
}

/// Splice a rewrite of every match into `current`.
///
/// Applied from the back, so an edit near the end of the file cannot move the
/// byte offsets of one still to be applied. Every span is checked against the
/// text first, and one bad span refuses the whole file rather than producing a
/// half-rewritten file that still parses.
fn splice_replacements(
    current: &str,
    spans: &[(usize, usize, &str)],
    display: &str,
) -> Result<String, String> {
    for (start, end, _) in spans {
        if *end > current.len()
            || start > end
            || !current.is_char_boundary(*start)
            || !current.is_char_boundary(*end)
        {
            return Err(err(format!(
                "a match ast-grep reported in {display} does not line up with the file \
                 on disk, so nothing was changed"
            )));
        }
    }
    let mut ordered: Vec<&(usize, usize, &str)> = spans.iter().collect();
    ordered.sort_by_key(|span| std::cmp::Reverse(span.0));
    let mut updated = current.to_string();
    for (start, end, text) in ordered {
        updated.replace_range(*start..*end, text);
    }
    Ok(updated)
}

/// Files git already reports as modified, among the ones asked about.
///
/// CI-4's recorded trap is applying a change across the whole repository when
/// the tree is already dirty, because afterwards nobody can tell which lines the
/// codemod wrote and which were already there. Refusing is the wrong answer —
/// an agent's own previous edits are uncommitted by definition, so that would
/// make the tool unusable exactly when it is wanted. The answer is to make the
/// separation visible instead: the diff this tool shows is its own change and
/// nothing else, and it names every file that was already modified so the
/// entanglement is on the record rather than discovered later.
fn already_modified(root: &Path, files: &[String]) -> Vec<String> {
    if files.is_empty() {
        return Vec::new();
    }
    let mut command = std::process::Command::new("git");
    command.current_dir(root).arg("status").arg("--porcelain");
    for file in files {
        command.arg("--").arg(file);
    }
    let Ok(output) = command.output() else {
        // Not a repository, or git is missing. Not a reason to refuse: the tool's
        // own diff is still exactly its own change.
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.get(3..).map(|path| path.trim().to_string()))
        .collect()
}

/// Run ast-grep in `scan` mode over a rule the caller wrote, and return matches.
///
/// A rule ast-grep cannot parse exits 8 with a message on stderr, which is a
/// fact about the rule and is reported as one. A rule that parses and matches
/// nothing exits 1 with an empty list, which is *not* a fact about the code and
/// the caller is told so.
fn run_ast_grep_rule(
    root: &Path,
    rule: &str,
    target: &str,
    timeout_secs: u64,
) -> Result<Vec<AstMatch>, String> {
    let binary = ast_grep_binary()?;
    let mut command = std::process::Command::new(&binary);
    command.current_dir(root);
    command
        .arg("scan")
        .arg("--inline-rules")
        .arg(rule)
        .arg("--json")
        .arg(target);
    let output = run_with_timeout(&mut command, timeout_secs)
        .map_err(|e| err(format!("could not run {binary}: {e}")))?;
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if output.status.code() == Some(8) || stderr.starts_with("Error: Cannot parse rule") {
        return Err(err(format!(
            "ast-grep could not read the rule, so nothing was searched and no file was \
             touched. A rule needs an `id`, a `language`, a `rule:` block with a `pattern`, \
             and a `fix:` block to change anything. ast-grep said: {stderr}"
        )));
    }
    if !output.status.success() && !stderr.is_empty() {
        return Err(err(format!("ast-grep failed: {stderr}")));
    }
    parse_ast_matches(&String::from_utf8_lossy(&output.stdout))
}

/// What a `codemod` call found.
#[derive(Debug)]
pub struct CodemodPlan {
    sites: Vec<String>,
    rewrites: Vec<PlannedRewrite>,
    /// False when the rule carries no `fix:`, so it can only report.
    rewrites_code: bool,
    /// Of the files to be rewritten, those git already reports as modified.
    entangled: Vec<String>,
}

/// Work out what a `codemod` call would do, without touching a file.
pub fn plan_codemod(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> Result<CodemodPlan, String> {
    let Some(rule) = arg_str(args, "rule") else {
        return Err(err("codemod needs a string \"rule\""));
    };
    if rule.trim().is_empty() {
        return Err(err("codemod needs a non-empty \"rule\""));
    }
    let target = arg_str(args, "path").unwrap_or(".").trim().to_string();
    if target.is_empty() {
        return Err(err("codemod needs a non-empty \"path\""));
    }
    let (full, display) = workspace_path(root, &target)?;
    if !full.exists() {
        return Err(err(format!("{display} does not exist")));
    }

    let matches = run_ast_grep_rule(root, rule, &display, timeout_secs)?;
    if matches.is_empty() {
        return Err(err(format!(
            "codemod matched no sites under {display}, and nothing was changed. That \
             result cannot distinguish a rule whose pattern is wrong from code that does \
             not contain it. Check the pattern's metavariables are written $NAME, and \
             that the shape is present, before reading this as \"the code is already \
             correct\"."
        )));
    }

    let sites = describe_sites(&matches);
    // A rule with no `fix:` is a report, and ast-grep says so by leaving the
    // replacement out of every match rather than by failing.
    let rewrites_code = matches.iter().any(|m| m.replacement.is_some());
    let rewrites = if rewrites_code {
        plan_file_rewrites(root, &matches, ToolName::Codemod)?
    } else {
        Vec::new()
    };
    let touched: Vec<String> = rewrites.iter().map(|r| r.display.clone()).collect();
    let entangled = already_modified(root, &touched);

    Ok(CodemodPlan {
        sites,
        rewrites,
        rewrites_code,
        entangled,
    })
}

/// The per-file section the preview and the result both print.
///
/// Reads from the plan, so it is the same diff in both places: the preview and
/// the outcome cannot describe different edits.
fn describe_rewrites(rewrites: &[PlannedRewrite]) -> String {
    let mut out = String::new();
    for rewrite in rewrites {
        out.push_str(&format!(
            "\n\ntarget: {} ({} site(s))\n{}",
            rewrite.display,
            rewrite.sites,
            unified_diff(&rewrite.current, &rewrite.updated).trim_end()
        ));
    }
    out
}

/// Write a plan's rewrites, one file at a time, atomically.
///
/// Each file is whole before and after, so a failure part-way through leaves the
/// files already written correct and the rest untouched, and the caller is told
/// which file stopped it rather than being handed a partial success.
fn write_rewrites(rewrites: &[PlannedRewrite]) -> Result<(), String> {
    for rewrite in rewrites {
        xencode_core_rs::write_atomic(&rewrite.full, rewrite.updated.as_bytes())
            .map_err(|e| err(format!("cannot write {}: {e}", rewrite.display)))?;
    }
    Ok(())
}

/// `codemod`: apply one ast-grep rule across the tree, or report what it hits.
fn tool_codemod(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> String {
    let plan = match plan_codemod(root, args, timeout_secs) {
        Ok(plan) => plan,
        Err(reason) => return reason,
    };
    if !plan.rewrites_code {
        // Counted from the sites, not the rewrites: there are no rewrites in this
        // mode, and "0 site(s)" beside a list of two would be its own small lie.
        return format!(
            "codemod: the rule has no `fix:`, so it reports and changes nothing — \
             {} site(s):\n{}",
            plan.sites.len(),
            plan.sites.join("\n")
        );
    }

    if let Err(reason) = write_rewrites(&plan.rewrites) {
        return reason;
    }
    let total: usize = plan.rewrites.iter().map(|r| r.sites).sum();
    let mut out = format!(
        "codemod: rewrote {total} site(s) across {} file(s):\n{}",
        plan.rewrites.len(),
        describe_rewrites(&plan.rewrites)
    );
    if !plan.entangled.is_empty() {
        out.push_str(&format!(
            "\n(note: {} of those file(s) already had uncommitted changes before this \
             ran — {} — so a git revert or /rewind of those files takes the earlier edits \
             with it. The diff above is this rule's change only.)",
            plan.entangled.len(),
            plan.entangled.join(", ")
        ));
    }
    out.trim_end().to_string()
}

/// What a `rename` call resolved, before anything is written.
#[derive(Debug)]
pub(crate) struct RenamePlan {
    /// The symbol being renamed.
    pub symbol: String,
    /// Its replacement.
    pub new_name: String,
    /// Root-relative file holding the single definition.
    pub definition: String,
    /// What kind of item it is (`function`, `struct`, `enum`, `trait`).
    pub kind: String,
    /// Per-file rewrites, definition site included.
    pub rewrites: Vec<PlannedRewrite>,
    /// How the sites were found, said in the preview and in the result, because
    /// the two ways differ in what they can rename by mistake.
    pub basis: String,
}

/// The rename from rust-analyzer's semantic index (`LSP-3`): exactly the places
/// that resolve to the one symbol, so a struct field, a local variable or an
/// unrelated item that happens to share the name is left alone.
///
/// `Ok(None)` means there is no current index to answer from, with the reason,
/// and the caller falls back to the syntactic rename. `Err` is a refusal.
fn semantic_rename_plan(
    root: &Path,
    symbol: &str,
    new_name: &str,
    in_file: Option<&str>,
) -> Result<Result<RenamePlan, String>, String> {
    use xencode_context_rs::scip_index;
    let Ok(workspace) = xencode_context_rs::verify::manifest_dir(root) else {
        return Ok(Err("there is no Cargo workspace here".to_string()));
    };
    let found = match scip_index::symbols_named(&workspace, symbol) {
        Ok(found) => found,
        Err(why) => return Ok(Err(why.to_string())),
    };
    // Paths in the index are workspace-relative; the rest of this file speaks
    // root-relative, so the workspace's place under the root is put in front.
    let prefix = match workspace.strip_prefix(root) {
        Ok(rel) if rel.as_os_str().is_empty() => String::new(),
        Ok(rel) => format!("{}/", rel.to_string_lossy().replace('\\', "/")),
        Err(_) => {
            return Ok(Err(
                "the Cargo workspace is not inside the project root".to_string()
            ))
        }
    };
    let candidates: Vec<&scip_index::NamedSymbol> = match in_file {
        Some(asked) => {
            let asked = asked.trim().trim_start_matches("./").replace('\\', "/");
            found
                .iter()
                .filter(|n| {
                    n.defined_in.iter().any(|f| {
                        let full = format!("{prefix}{f}");
                        full == asked || full.ends_with(&format!("/{asked}"))
                    })
                })
                .collect()
        }
        None => found.iter().collect(),
    };
    let target = match candidates.as_slice() {
        [] if found.is_empty() => {
            return Err(err(format!(
                "rust-analyzer's index has no declaration of `{symbol}` in this \
                 workspace, so there is nothing to rename. A local variable is not \
                 renamed by name, and an item from another crate cannot be"
            )))
        }
        [] => {
            return Err(err(format!(
                "`{symbol}` is declared in {}, not in {}",
                found
                    .iter()
                    .flat_map(|n| n.defined_in.iter().map(|f| format!("{prefix}{f}")))
                    .collect::<Vec<_>>()
                    .join(", "),
                in_file.unwrap_or_default()
            )))
        }
        [one] => *one,
        several => {
            let list = several
                .iter()
                .map(|n| {
                    format!(
                        "a {} in {}",
                        n.kind,
                        n.defined_in
                            .iter()
                            .map(|f| format!("{prefix}{f}"))
                            .collect::<Vec<_>>()
                            .join(" and ")
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            return Err(err(format!(
                "`{symbol}` names {} different items — {list} — so rename refuses \
                 rather than guess. Pass \"in\" with the file that declares the one \
                 you mean",
                several.len()
            )));
        }
    };

    let mut files: Vec<&str> = Vec::new();
    for site in &target.sites {
        if !files.contains(&site.file.as_str()) {
            files.push(&site.file);
        }
    }
    let mut rewrites = Vec::new();
    for file in files {
        let (full, display) = workspace_path(root, &format!("{prefix}{file}"))?;
        let current = read_text(&full, &display)?;
        let mut spans: Vec<(usize, usize, &str)> = Vec::new();
        for site in target.sites.iter().filter(|s| s.file == file) {
            let checked = scip_index::byte_range(&current, site)
                .filter(|(s, e)| current.get(*s..*e) == Some(symbol));
            let Some((start, end)) = checked else {
                return Err(err(format!(
                    "{display}:{} does not hold `{symbol}` where rust-analyzer's index \
                     put it, so nothing was renamed. The file has changed since the \
                     index was built",
                    site.line + 1
                )));
            };
            spans.push((start, end, new_name));
        }
        let updated = splice_replacements(&current, &spans, &display)?;
        rewrites.push(PlannedRewrite {
            full,
            display,
            current,
            updated,
            sites: spans.len(),
        });
    }
    Ok(Ok(RenamePlan {
        symbol: symbol.to_string(),
        new_name: new_name.to_string(),
        definition: target
            .defined_in
            .iter()
            .map(|f| format!("{prefix}{f}"))
            .collect::<Vec<_>>()
            .join(", "),
        kind: target.kind.to_string(),
        rewrites,
        basis: "Sites from rust-analyzer's semantic index: every place that resolves to \
                this one item, and nothing else that shares its name."
            .to_string(),
    }))
}

/// Whether a name can be a Rust identifier. Shape only — `cargo check` after
/// the rewrite is what judges the rest.
fn is_ident(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Rust's strict keywords: renaming onto one of these always breaks the build,
/// so it is refused with the list rather than discovered via `cargo check`.
fn is_keyword(name: &str) -> bool {
    matches!(
        name,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
            | "try"
    )
}

/// Definitions of `symbol` across the workspace's Rust files, as
/// (root-relative file, item kind). Uses the tree-sitter tier, not text
/// search, so a mention in a comment is never mistaken for a declaration.
fn definition_sites(root: &Path, symbol: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path.extension().map(|e| e != "rs").unwrap_or(true) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let symbols = xencode_context_rs::extract_tree_symbols(&text);
            let kind = if symbols.functions.iter().any(|f| f == symbol) {
                Some("function")
            } else if symbols.structs.iter().any(|f| f == symbol) {
                Some("struct")
            } else if symbols.enums.iter().any(|f| f == symbol) {
                Some("enum")
            } else if symbols.traits.iter().any(|f| f == symbol) {
                Some("trait")
            } else {
                None
            };
            if let Some(kind) = kind {
                let display = path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_default();
                out.push((display, kind.to_string()));
            }
        }
    }
    out.sort();
    out
}

/// Work out what a `rename` call would do, without touching a file.
///
/// Resolution first, rewriting second: the definition is found through the
/// symbol index, and only a uniquely-defined symbol proceeds. References are
/// then rewritten through ast-grep's identifier matches, definition site
/// included, so the declaration and every use move together.
pub(crate) fn plan_rename(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> Result<RenamePlan, String> {
    let Some(symbol) = arg_str(args, "symbol") else {
        return Err(err("rename needs a string \"symbol\""));
    };
    if symbol.trim().is_empty() {
        return Err(err("rename needs a non-empty \"symbol\""));
    }
    let Some(new_name) = arg_str(args, "new_name") else {
        return Err(err("rename needs a string \"new_name\""));
    };
    if !is_ident(new_name) {
        return Err(err(format!(
            "{new_name:?} is not a Rust identifier, so nothing was renamed"
        )));
    }
    if is_keyword(new_name) {
        return Err(err(format!(
            "{new_name:?} is a Rust keyword, so renaming onto it would break the build"
        )));
    }
    let in_file = arg_str(args, "in").filter(|s| !s.trim().is_empty());

    // The semantic index first (LSP-3); without a current one, the syntactic
    // rename below, which says why it was used.
    let fallback_reason = match semantic_rename_plan(root, symbol, new_name, in_file)? {
        Ok(plan) => return Ok(plan),
        Err(why) => why,
    };

    let definitions = definition_sites(root, symbol);
    let (definition, kind) = match definitions.as_slice() {
        [] => {
            return Err(err(format!(
                "no definition of `{symbol}` in the workspace's Rust files, so there is                  nothing to rename. A use without a declaration is not renamed, because                  the tool cannot know which item the name belongs to"
            )));
        }
        [(file, kind)] => (file.clone(), kind.clone()),
        several => {
            let list = several
                .iter()
                .map(|(f, k)| format!("{f} ({k})"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(err(format!(
                "`{symbol}` is defined in more than one place — {list} — so rename                  refuses rather than guess which one was meant. Narrow it with an                  `edit_symbol` call on the right file instead"
            )));
        }
    };

    // A bare identifier pattern matches identifier nodes, not substrings, so
    // `foo` renames `foo` and never `foobar`. The rewrite text is the new name
    // at every site, including the definition found above.
    let matches = run_ast_grep(
        root,
        symbol,
        ".",
        Some(new_name),
        Some("rust"),
        timeout_secs,
    )?;
    if matches.is_empty() {
        return Err(err(format!(
            "the index declares `{symbol}` in {definition} but ast-grep finds no              identifier sites for it, so nothing was renamed. The declaration may              use a form the pattern cannot see"
        )));
    }
    let rewrites = plan_file_rewrites(root, &matches, ToolName::Rename)?;
    Ok(RenamePlan {
        symbol: symbol.to_string(),
        new_name: new_name.to_string(),
        definition,
        kind,
        rewrites,
        basis: format!(
            "Sites from ast-grep: every Rust identifier spelled `{symbol}`, which also \
             catches a field, a local variable or another item with that name. Not from \
             rust-analyzer's semantic index: {fallback_reason}."
        ),
    })
}

fn tool_rename(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> String {
    let plan = match plan_rename(root, args, timeout_secs) {
        Ok(plan) => plan,
        Err(reason) => return reason,
    };
    if let Err(reason) = write_rewrites(&plan.rewrites) {
        return reason;
    }
    let total: usize = plan.rewrites.iter().map(|r| r.sites).sum();
    let mut out = format!(
        "rename: `{}` ({} in {}) became `{}` at {total} site(s) across {} file(s):\n{}",
        plan.symbol,
        plan.kind,
        plan.definition,
        plan.new_name,
        plan.rewrites.len(),
        describe_rewrites(&plan.rewrites)
    );
    out.push_str(&format!("\n{}", plan.basis));
    // The second half of QI-2's substrate: the rewrite is checked, not assumed.
    // A failure is reported with the errors, not reverted — the preview gate
    // approved the diff, and silent reverts destroy the evidence of what broke.
    match cargo_check_status(root, timeout_secs) {
        Ok(()) => out.push_str("\n`cargo check` passes on the renamed tree."),
        Err(errors) => out.push_str(&format!(
            "\n`cargo check` FAILS on the renamed tree, so the rename is incomplete:\n{errors}"
        )),
    }
    out.trim_end().to_string()
}

/// Whether `cargo check` passes in the workspace holding `root`.
fn cargo_check_status(root: &Path, timeout_secs: u64) -> Result<(), String> {
    let manifest = xencode_context_rs::verify::manifest_dir(root)
        .map_err(|e| format!("could not locate the manifest: {e}"))?;
    let mut command = std::process::Command::new("cargo");
    command
        .current_dir(&manifest)
        .arg("check")
        .arg("--all-targets");
    match run_with_timeout(&mut command, timeout_secs.max(60)) {
        Err(e) => Err(format!("could not run cargo check: {e}")),
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let errors: Vec<&str> = stderr
                .lines()
                .filter(|l| l.starts_with("error"))
                .take(8)
                .collect();
            Err(if errors.is_empty() {
                format!("exit {}", output.status.code().unwrap_or(-1))
            } else {
                errors.join("\n")
            })
        }
    }
}

/// `ast_edit`: report the sites, and rewrite them when a replacement was given.
fn tool_ast_edit(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
) -> String {
    let plan = match plan_ast_edit(root, args, timeout_secs) {
        Ok(plan) => plan,
        Err(reason) => return reason,
    };
    let pattern = arg_str(args, "pattern").unwrap_or_default();

    if plan.replacement.is_none() {
        return format!(
            "ast_edit: {} site(s) match `{pattern}`, nothing changed:\n{}",
            plan.sites.len(),
            plan.sites.join("\n")
        );
    }

    let total: usize = plan.rewrites.iter().map(|r| r.sites).sum();
    let mut out = format!(
        "ast_edit: rewrote {total} site(s) matching `{pattern}` across {} file(s):\n",
        plan.rewrites.len()
    );
    if let Err(reason) = write_rewrites(&plan.rewrites) {
        return reason;
    }
    for rewrite in &plan.rewrites {
        out.push_str(&format!(
            "\n{} ({} site(s))\n{}",
            rewrite.display,
            rewrite.sites,
            unified_diff(&rewrite.current, &rewrite.updated).trim_end()
        ));
    }
    out.trim_end().to_string()
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
///
/// A plain `cargo build` or `cargo check` is asked for rustc's machine-readable
/// output instead of its rendered text, and the answer is rebuilt from that —
/// error code, position, the fix the compiler offers, and the error-index entry
/// for the code. See [`xencode_core_rs::rustc_json`].
async fn run_foreground(
    root: &Path,
    command: &str,
    timeout_secs: u64,
    sandbox: &crate::sandbox::Sandbox,
    net: bool,
) -> String {
    let json_form = xencode_core_rs::cargo_json_command(command);
    let asked_for = json_form.as_deref().unwrap_or(command);
    // With the sandbox on this becomes `bwrap … sh -c`; off it is the plain
    // shell; asked-for but impossible it is an error, never a quiet pass.
    let (program, args) = match sandbox.wrap(asked_for, net) {
        Ok(Some(wrapped)) => wrapped,
        Ok(None) => (
            "sh".to_string(),
            vec!["-c".to_string(), asked_for.to_string()],
        ),
        Err(reason) => return err(reason),
    };
    let child = match tokio::process::Command::new(&program)
        .args(&args)
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
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Cargo writes the JSON to stdout and keeps its own progress and summary on
    // stderr, so the summary stays and the JSON is what gets read.
    let body = match json_form
        .as_deref()
        .and_then(|_| xencode_core_rs::parse_rustc_json(&stdout))
    {
        Some(report) => format!("{}\n{}", report.render(), stderr.trim()),
        None => format!("{stdout}{stderr}"),
    };
    let body = body.trim();
    let (dropped, tail) = cap_tail(body, COMMAND_OUTPUT_CAP);
    let status = match output.status.code() {
        Some(code) => format!("exit {code}"),
        None => "killed by signal".to_string(),
    };
    let mut result = format!("$ {asked_for}\n{status}");
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

/// One of the two hook phases (I3-02).
#[derive(Clone, Copy, PartialEq, Eq)]
enum HookPhase {
    Before,
    After,
}

impl HookPhase {
    fn word(self) -> &'static str {
        match self {
            HookPhase::Before => "before",
            HookPhase::After => "after",
        }
    }
}

/// The hook declared for a tool call's exact name, else the `*` catch-all.
/// Returns the configured key and the command, so the transcript names what
/// the user actually wrote.
fn hook_for<'a>(
    hooks: &'a xencode_config_rs::AgentHooks,
    phase: HookPhase,
    tool: &'a str,
) -> Option<(&'a str, &'a str)> {
    let table = match phase {
        HookPhase::Before => &hooks.before,
        HookPhase::After => &hooks.after,
    };
    table
        .get(tool)
        .map(|command| (tool, command.as_str()))
        .or_else(|| table.get("*").map(|command| ("*", command.as_str())))
}

/// Run one hook command — `sh -c` in the workspace root on the same budget
/// and output cap as `run_command`. The bool says whether it exited clean;
/// the string is always an annotated line (status first, output tail only on
/// a non-zero exit), so a capped or empty body can never look like success.
///
/// The event is delivered as JSON on the hook's **stdin**, never in the command
/// line: a `tool_input` can carry a file's whole contents, and anything in argv
/// is readable through `/proc` by any local process. The shape follows the
/// event names the wider ecosystem already uses (`hook_event_name`,
/// `tool_name`, `tool_input`, `cwd`, `session_id`) so a hook written for another
/// agent runs here unchanged — xencode adds no fourth dialect.
async fn run_hook(
    root: &Path,
    name: &str,
    call: &ToolCall,
    phase: HookPhase,
    command: &str,
    ctx: &ApprovalCtx,
) -> (bool, String) {
    let tool = call.name.as_str();
    let payload = serde_json::json!({
        "hook_event_name": match phase {
            HookPhase::Before => "PreToolUse",
            HookPhase::After => "PostToolUse",
        },
        "tool_name": tool,
        "tool_input": call.arguments_object(),
        "cwd": root.to_string_lossy(),
        "session_id": ctx.session_id.as_deref().unwrap_or(""),
    });
    let payload = serde_json::to_vec(&payload).unwrap_or_else(|_| Vec::new());

    // SE-7: a hook is a shell command the config named, so it is exactly the
    // exfiltration path the sandbox exists to bound. It gets no per-call net
    // grant — a hook runs with the network off when the sandbox is on.
    let (program, hook_args) = match ctx.sandbox.wrap(command, false) {
        Ok(Some(wrapped)) => wrapped,
        Ok(None) => (
            "sh".to_string(),
            vec!["-c".to_string(), command.to_string()],
        ),
        Err(reason) => return (false, format!("hook[{name}] {}: {reason}", phase.word())),
    };
    let mut child = match tokio::process::Command::new(&program)
        .args(&hook_args)
        .current_dir(root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            return (
                false,
                format!(
                    "hook[{name}] {} {tool}: cannot run {command:?}: {e}",
                    phase.word()
                ),
            );
        }
    };
    // Hand the event over, then drop the writer so a hook that reads to EOF is
    // not left hanging. A hook that ignores stdin never reads it; the closed
    // pipe is simply ignored. A child whose stdin already closed (it exited)
    // makes this write fail, which is not the hook's fault and must not mask its
    // exit status — so the write error is swallowed and the wait below reports
    // the real outcome.
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(&payload).await;
        let _ = stdin.shutdown().await;
    }
    let secs = ctx.command_timeout.max(1);
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(secs),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return (
                false,
                format!(
                    "hook[{name}] {} {tool}: failed to run {command:?}: {e}",
                    phase.word()
                ),
            );
        }
        Err(_) => {
            return (
                false,
                format!(
                    "hook[{name}] {} {tool}: timed out after {secs}s and was killed",
                    phase.word()
                ),
            );
        }
    };
    let body = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let body = body.trim();
    let (dropped, tail) = cap_tail(body, COMMAND_OUTPUT_CAP);
    let ok = output.status.success();
    let status = match output.status.code() {
        Some(code) => format!("exit {code}"),
        None => "killed by signal".to_string(),
    };
    let mut line = format!("hook[{name}] {} {tool}: {status}", phase.word());
    if dropped {
        line.push_str(&format!(
            "\n(output capped to the last {} bytes of {})",
            COMMAND_OUTPUT_CAP,
            body.len()
        ));
    }
    if !tail.is_empty() {
        line.push('\n');
        line.push_str(tail);
    }
    (ok, line)
}

pub async fn execute_tool_call(rt: &TaskRuntime, root: &Path, call: &ToolCall) -> String {
    execute_tool_call_timed(rt, root, call, DEFAULT_COMMAND_TIMEOUT).await
}

/// [`execute_tool_call`] with the caller's configured foreground timeout.
///
/// These two entry points are the ones outside the chat loop: they read crate
/// documentation offline only, and never fetch the web at all, because the two
/// settings that permit those trips arrive through [`ApprovalCtx`]; they search
/// on no engine, for the same reason — the provider is a config value and this
/// path reads no config; they know no skills, because the session's loaded skills
/// arrive the same way; and they carry no reproduction gate, because the session's
/// gate does too. The loop is the only caller that has any of them.
pub async fn execute_tool_call_timed(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    command_timeout: u64,
) -> String {
    execute_tool_call_plan(
        rt,
        root,
        call,
        command_timeout,
        None,
        None,
        false,
        false,
        &Ok(xencode_analysis_rs::SearchProvider::None),
        None,
        &crate::sandbox::Sandbox::disabled(),
        None,
        None,
    )
    .await
}

/// The dispatcher. `plan` is the chat's visible todo list: only the loop has
/// one, so `update_plan` outside it is an error rather than a silent no-op.
#[allow(clippy::too_many_arguments)] // each argument is one session surface the loop alone has
async fn execute_tool_call_plan(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    command_timeout: u64,
    plan: Option<&PlanHandle>,
    mcp: Option<&crate::mcp::McpHub>,
    online_docs: bool,
    web_fetch: bool,
    search: &Result<xencode_analysis_rs::SearchProvider, String>,
    skills: Option<&xencode_plugin_rs::SkillRuntime>,
    sandbox: &crate::sandbox::Sandbox,
    repro: Option<&crate::reprogate::ReproGate>,
    session_id: Option<&str>,
) -> String {
    let args = call.arguments_object();
    // Server tools are addressed by their visible `mcp__<server>__<tool>` name;
    // the hub knows which real tool that stands for.
    if crate::mcp::is_mcp_tool(&call.name) {
        return match mcp {
            Some(hub) => {
                hub.call(&call.name, serde_json::Value::Object(args.clone()))
                    .await
            }
            None => err("MCP tools are only available in the chat loop"),
        };
    }
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
                // SE-7: the per-call `net` grant lifts the network isolation
                // the sandbox imposes; it is ignored when the sandbox is off.
                let net = args.get("net").and_then(|v| v.as_bool()).unwrap_or(false);
                run_foreground(root, command, command_timeout, sandbox, net).await
            }
            _ => "error: run_command needs a non-empty string \"command\"".to_string(),
        },
        // U-6: run the reproduction, judge it, and move the gate's phase. The
        // gate is the session's, so the measurement survives the turn that made
        // it; outside the chat loop there is no gate to write into.
        "reproduce_bug" => match repro {
            Some(gate) => {
                crate::reprogate::reproduce_bug(
                    gate,
                    root,
                    &args,
                    command_timeout,
                    sandbox,
                    session_id,
                )
                .await
            }
            None => err("reproduce_bug is only available in the chat loop"),
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
            let net = args.get("net").and_then(|v| v.as_bool()).unwrap_or(false);
            // SE-7: a background task is isolated the same way a foreground one
            // is — the recorded command stays readable, the program actually
            // spawned is the sandboxed one. An enabled-but-impossible sandbox
            // refuses rather than falling back quietly.
            let spec = match sandbox.wrap(command, net) {
                Ok(Some((program, wrap_args))) => xencode_core_rs::tasks::SpawnSpec {
                    program,
                    args: wrap_args,
                },
                Ok(None) => xencode_core_rs::tasks::SpawnSpec::shell(command),
                Err(reason) => return err(reason),
            };
            let mut m = rt.lock().await;
            match m
                .start_spawning(
                    name,
                    command,
                    cwd.as_deref(),
                    xencode_core_rs::tasks::DEFAULT_TASK_TIMEOUT,
                    &spec,
                )
                .await
            {
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
        "read_docs" => tool_read_docs(root, &args, online_docs).await,
        "lookup_advisory" => tool_lookup_advisory(root, &args),
        // RS-1. Off, this is not an error the model can retry around: the tool
        // is not offered at all, so reaching here means a caller that never
        // asked the user. On, it still cannot choose its own destination —
        // `fetch_url_guarded` is the only entry point used, whatever the mode.
        "web_fetch" if !web_fetch => err(
            "web_fetch is not enabled: `xencode config set allow_web_fetch true` offers it, \
                 and every call still asks before the request leaves",
        ),
        "web_fetch" => tool_web_fetch(&args).await,
        // RS-2: the same shape as the fetch above. A model that was never shown
        // the tool cannot run it, and a config that names an engine it cannot use
        // yet — `searxng` with no address, a paid key that is missing — is answered
        // with the setting to change rather than a transport failure.
        "web_search" if matches!(search, Ok(xencode_analysis_rs::SearchProvider::None)) => err(
            "web_search is not enabled: `xencode config set search_provider wikipedia` points \
                 it at an engine, `searxng` at one you run yourself, and every call still asks \
                 before the question leaves",
        ),
        "web_search" => match search {
            Err(reason) => err(reason.clone()),
            Ok(provider) => tool_web_search(provider, &args).await,
        },
        "load_skill" => tool_load_skill(skills, &args),
        "write_file" => tool_write_file(root, &args),
        "write_note" => tool_write_note(root, &args),
        "edit_file" => tool_edit_file(root, &args),
        "edit_symbol" => tool_edit_symbol(root, &args),
        "ast_edit" => tool_ast_edit(root, &args, command_timeout),
        "codemod" => tool_codemod(root, &args, command_timeout),
        "rename" => tool_rename(root, &args, command_timeout),
        "what_breaks" => tool_what_breaks(root, &args),
        "debug_test" => tool_debug_test(root, &args).await,
        "find_refs" => tool_find_refs(root, &args, false),
        "callers" => tool_find_refs(root, &args, true),
        other => format!("error: unknown tool {other}"),
    }
}

/// Whether a basename is secret-named (SE-4): private keys, credential
/// stores, and dotenv files. Matched on the name alone, because the gate
/// judges the call before it runs and cannot read the content first.
fn secret_basename(name: &str) -> bool {
    let name = name.to_lowercase();
    name.starts_with("id_rsa")
        || name.starts_with("id_ed25519")
        || name.starts_with("id_ecdsa")
        || name.starts_with("id_dsa")
        || name.ends_with(".pem")
        || name.ends_with(".key")
        || name.ends_with(".p12")
        || name.ends_with(".pfx")
        || name == ".env"
        || name.starts_with(".env.")
        || name.contains("credential")
        || name.contains("secret")
        || name.contains("passwd")
}

/// Whether a path reaches secrets: under `~/.ssh` or `~/.xencode`, or
/// secret-named itself. Lexical only — existence is irrelevant, and touching
/// the filesystem to judge a path the gate may refuse would be backwards.
/// `home` is passed in so tests never read the real one.
pub fn sensitive_path(home: Option<&Path>, root: &Path, path: &Path) -> bool {
    let absolute = normalize(&if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    });
    if absolute
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(secret_basename)
    {
        return true;
    }
    let Some(home) = home else {
        return false;
    };
    [".ssh", ".xencode"].iter().any(|dir| {
        let base = normalize(&home.join(dir));
        absolute.starts_with(&base)
    })
}

/// Whether a shell command dumps the environment by design (`env`,
/// `printenv`, bare `set`, `export -p`, each optionally under one `sudo`).
/// The output still has to carry secrets to matter — this only names the
/// commands whose whole point is printing them — but a bare `env` in a
/// session taints on intent, because waiting for the output to prove it
/// would taint one call too late.
fn env_dump_command(command: &str) -> bool {
    let mut words = command.split_whitespace();
    let mut first = words.next().unwrap_or("");
    if first == "sudo" {
        first = words.next().unwrap_or("");
    }
    match first {
        "env" | "printenv" => true,
        "set" => words.next().is_none(),
        "export" => words.next() == Some("-p"),
        _ => false,
    }
}

/// What the loop needs to gate a call before running it: the configured
/// mode, the session grants — shared with `App` so an "always allow" answer
/// is still in force for the next message — and the channel the approval
/// overlay drains each frame.
#[derive(Clone)]
pub struct ApprovalCtx {
    pub mode: ApprovalMode,
    /// Explicit headless policy gating calls when running headlessly (AE-7).
    pub headless_policy: Option<HeadlessPolicy>,
    pub grants: Arc<std::sync::Mutex<Vec<ToolClass>>>,
    pub prompts: mpsc::UnboundedSender<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
    /// Where writes get snapshotted before they land (I2-01 checkpoints).
    pub checkpoints: Arc<CheckpointStore>,
    /// Which turn's group this loop records into.
    pub turn: usize,
    /// Wall-clock budget for `run_command` (`agent_command_timeout`).
    pub command_timeout: u64,
    /// The SE-7 `bwrap` sandbox decision for this run: `run_command`,
    /// `background_start` and shell hooks are wrapped when it says so. Built
    /// once per run by `App::approval_ctx` from the config plus whether this
    /// machine has `bwrap`; a run outside the chat loop passes a disabled one.
    pub sandbox: crate::sandbox::Sandbox,
    /// The chat pane's todo list, written by `update_plan`.
    pub plan: PlanHandle,
    /// The session's started MCP servers (I3-01): the turn offers their tools
    /// and routes approved calls back to them. Empty until `/mcp` connects.
    pub mcp: Arc<crate::mcp::McpHub>,
    /// The skills this session loaded (M-3): the prompt carries their menu and
    /// `load_skill` reads one's instructions out of here. Empty is the default
    /// and offers no tool at all.
    pub skills: Arc<xencode_plugin_rs::SkillRuntime>,
    /// Pre/post shell hooks (I3-02): matched to each approved call by tool
    /// name. Empty by default — no hooks, no change in behavior.
    pub hooks: xencode_config_rs::AgentHooks,
    /// How each tool the model was offered describes its arguments, by name
    /// (MI-1). The loop fills this from the definitions it sent, so a call whose
    /// arguments do not match that description is answered with the mismatch
    /// rather than run as if it had asked for nothing. Empty means nothing is
    /// known, which checks the arguments that cannot be read at all and no more.
    pub schemas: std::collections::HashMap<String, serde_json::Value>,
    /// Whether `read_docs` may fetch when this machine has no copy of the
    /// version asked for (`allow_online_docs`). Off by default, and off for
    /// every path that is not the chat loop: a call out of the machine has to
    /// be something the user decided, not something the agent worked around.
    pub online_docs: bool,
    /// Whether `web_fetch` exists for this run at all (`allow_web_fetch`). A
    /// second guard behind the offer: the tool is not in the list a model is
    /// shown unless the user opened this, and a call that arrives from anywhere
    /// else — a resumed transcript, a caller that built its own request — is
    /// refused here rather than executed on the strength of having been named.
    pub web_fetch: bool,
    /// The search engine this run may use, resolved from the config when the run
    /// was built (RS-2). `Ok(SearchProvider::None)` means none is configured, and
    /// the tool is then not in the list the model is shown; `Err` carries the
    /// config's own words for the half that is missing, so a call that arrives
    /// from a resumed transcript is answered with what to fix rather than run on
    /// the strength of having been named.
    pub search: Result<xencode_analysis_rs::SearchProvider, String>,
    /// The session this turn belongs to, handed to hooks on stdin (M-1) so a
    /// hook can tell runs apart. `None` where no session is open.
    pub session_id: Option<String>,
    /// Every question this run asked a person and what they answered (QTR-5).
    /// Built fresh per run by `App::approval_ctx`, so the list a run's own tool
    /// tasks append to is that run's and nobody else's.
    pub approvals: Arc<std::sync::Mutex<Vec<xencode_context_rs::ApprovalRow>>>,
    /// Whether this session has touched secrets (SE-4). Shared across the
    /// session's runs like the grants above — deliberately one coarse bit,
    /// not per-variable taint, because cross-turn fine tracking is leaky and
    /// a leaky gate is theater. Once set it stays set for the session.
    pub taint: Arc<AtomicBool>,
    /// The secret values the context engine held back from this turn's dynamic
    /// tiers (PR-3), keyed by the placeholder the model was shown. A tool call
    /// that names a placeholder gets the real value back from here before it is
    /// classified and run, so the plaintext never crossed to the provider but
    /// the command still works. Empty/default when nothing was redacted.
    pub redaction: Arc<xencode_context_rs::Vault>,
    /// Where a ByteBot task's `ask_user` question goes (BT-2): the TUI shows
    /// it and sends the person's answer back. `None` everywhere else, and then
    /// the tool answers that it is only for ByteBot tasks.
    pub ask: Option<mpsc::UnboundedSender<(String, oneshot::Sender<String>)>>,
    /// The red-to-green reproduction gate (U-6). Shared across the session's
    /// runs like the taint bit, because its state is about the bug being fixed
    /// rather than the turn that noticed it: a gate that reset each turn would
    /// let turn two edit production on turn one's evidence. Off by default, and
    /// only the user opens it.
    pub repro: Arc<crate::reprogate::ReproGate>,
}

impl ApprovalCtx {
    fn granted(&self) -> Vec<ToolClass> {
        self.grants
            .lock()
            .map(|grants| grants.clone())
            .unwrap_or_default()
    }

    /// Whether `web_search` belongs in the list a model is shown (RS-2).
    ///
    /// The test is what the config *names*, not what it resolves to. A half-filled
    /// setting still offers the tool, because the alternative is that a person who
    /// typed `search_provider searxng` sees the tool quietly missing and no reason
    /// anywhere — asked that, it answers with the setting that is missing instead.
    /// Only `none` (or nothing at all) keeps the name out of the offer.
    pub fn search_offered(&self) -> bool {
        !matches!(self.search, Ok(xencode_analysis_rs::SearchProvider::None))
    }

    /// Whether this session has touched secrets (SE-4). Read at every
    /// classification, so a shell call after a secret read asks no matter
    /// what the mode or the session grants say.
    pub fn tainted(&self) -> bool {
        self.taint.load(Ordering::Relaxed)
    }

    /// Mark the session tainted. One way: exposure cannot be unseen, and a
    /// gate that forgets is a gate that exfiltrates on the second try.
    pub fn taint(&self) {
        self.taint.store(true, Ordering::Relaxed);
    }

    /// Inspect a finished tool call for secret exposure (SE-4) and taint the
    /// session when it finds any. Three shapes: a file tool touching a
    /// sensitive path, a command that dumps the environment by design, and
    /// secret-shaped text in anything a tool returned (a `cat` of a key
    /// reaches no sensitive path of its own, but its output is the key).
    /// Over-approximation is the design: taint buys a prompt, not a refusal,
    /// and a missed secret costs an exfiltration.
    pub fn note_tool_result(&self, root: &Path, call: &ToolCall, result: &str) {
        if self.tainted() {
            return;
        }
        let home = dirs::home_dir();
        let args = call.arguments_object();
        let path_touched = matches!(
            call.name.as_str(),
            "read_file" | "edit_file" | "write_file" | "list_dir" | "search_files"
        ) && ["path", "cwd"].iter().any(|key| {
            arg_str(&args, key)
                .is_some_and(|p| sensitive_path(home.as_deref(), root, Path::new(&p)))
        });
        let env_dumped = matches!(call.name.as_str(), "run_command" | "background_start")
            && arg_str(&args, "command").is_some_and(env_dump_command);
        if path_touched || env_dumped || xencode_context_rs::trace::contains_secret(result) {
            self.taint();
        }
    }

    fn grant(&self, class: ToolClass) {
        if let Ok(mut grants) = self.grants.lock() {
            if !grants.contains(&class) {
                grants.push(class);
            }
        }
    }

    /// Write one answered question into this run's ledger list (QTR-5). A lock
    /// that will not take is not worth failing a tool over: the call still runs
    /// or refuses as answered, and the row is the record, not the decision.
    fn record_approval(&self, tool: &str, class: ToolClass, answer: ApprovalAnswer) {
        let decision = match answer {
            ApprovalAnswer::Approved => xencode_context_rs::ApprovalDecision::Allowed,
            ApprovalAnswer::ApprovedForSession => {
                xencode_context_rs::ApprovalDecision::AlwaysAllowed
            }
            ApprovalAnswer::Denied => xencode_context_rs::ApprovalDecision::Denied,
        };
        if let Ok(mut rows) = self.approvals.lock() {
            rows.push(xencode_context_rs::ApprovalRow {
                tool: tool.to_string(),
                class: class.overlay_label().to_string(),
                decision,
            });
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

/// Run an approved call, checkpointing the target first. `mcp` is the session's
/// started servers: the loop has them, so a server tool outside it is an error
/// rather than a silent no-op (same shape as `plan`).
///
/// Hooks (I3-02) wrap this: a `before` hook runs first and — on a non-zero
/// exit — vetoes the call before any checkpoint is taken or anything runs;
/// an `after` hook runs last regardless of the call's outcome, its words
/// appended to the result so the model and the transcript both see them.
async fn run_and_checkpoint(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    ctx: &ApprovalCtx,
    mcp: Option<&crate::mcp::McpHub>,
) -> String {
    let mut before_note = String::new();
    if let Some((name, command)) = hook_for(&ctx.hooks, HookPhase::Before, &call.name) {
        let (ok, text) = run_hook(root, name, call, HookPhase::Before, command, ctx).await;
        if !ok {
            return err(format!("pre-hook vetoed this call:\n{text}"));
        }
        before_note.push_str(&text);
    }
    let note = ctx.snapshot_before(root, call);
    let mut result = execute_tool_call_plan(
        rt,
        root,
        call,
        ctx.command_timeout,
        Some(&ctx.plan),
        mcp,
        ctx.online_docs,
        ctx.web_fetch,
        &ctx.search,
        Some(&ctx.skills),
        &ctx.sandbox,
        Some(&ctx.repro),
        ctx.session_id.as_deref(),
    )
    .await;
    if let Some(note) = note {
        if !result.starts_with("error:") {
            result.push('\n');
            result.push_str(&note);
        }
    }
    if !before_note.is_empty() {
        result.insert_str(0, &format!("{before_note}\n"));
    }
    if let Some((name, command)) = hook_for(&ctx.hooks, HookPhase::After, &call.name) {
        let (_, text) = run_hook(root, name, call, HookPhase::After, command, ctx).await;
        result.push('\n');
        result.push_str(&text);
    }
    // SE-4: what just ran may have touched secrets — a key file, an env
    // dump, secret-shaped output. The session is tainted now if so, and
    // every later shell call asks, whatever the mode says.
    ctx.note_tool_result(root, call, &result);
    result
}

/// Tools whose reader deliberately accepts more shapes than the description
/// offered to the model claims — `update_plan` takes bare strings, checkbox
/// lines and invented key names because that is what small models write, and a
/// wrong plan is fixed by the next call rather than by a refusal. Enforcing the
/// description strictly on these would turn a working list into an error.
const LENIENT_READERS: &[&str] = &["update_plan"];

/// The loop's entry point (I1-04): policy first, prompt if the policy says
/// `Ask`, execute only on a yes. The result string is always something the
/// model can act on — a denial is stated as a denial.
pub async fn execute_tool_call_approved(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    ctx: &ApprovalCtx,
    mcp: Option<&crate::mcp::McpHub>,
) -> String {
    // BT-2: a ByteBot task asks the person a question. Nothing runs and
    // nothing is written, so there is no approval to ask for; the question
    // itself is the stop. The loop waits here until the person answers, types
    // "/done" after doing the step themselves, or cancels the task.
    if call.name == "ask_user" {
        let question = call
            .arguments_object()
            .get("question")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let Some(ask) = ctx.ask.as_ref() else {
            return "error: ask_user is only available to ByteBot tasks".to_string();
        };
        if question.is_empty() {
            return "error: ask_user needs a question".to_string();
        }
        let (reply, answer) = oneshot::channel();
        if ask.send((question, reply)).is_err() {
            return UNATTENDED_RESULT.to_string();
        }
        return match answer.await {
            Ok(text) if text.trim() == "/done" => {
                "The person did this step themselves. Continue from there.".to_string()
            }
            Ok(text) => format!("The person answered: {}", text.trim()),
            Err(_) => "The person did not answer; the task was cancelled.".to_string(),
        };
    }
    // A secret the context engine held back from this turn's dynamic tiers
    // reached the model only as a placeholder (PR-3). Here, at the point of
    // execution — after the provider saw the token, before the tool runs — the
    // placeholder becomes the real value again, so the shape check, the policy
    // classification and the run all work on what the call actually means. With
    // nothing redacted this is a no-op and `call` stays the borrowed original.
    let restored;
    let call = if ctx.redaction.is_empty() {
        call
    } else {
        restored = ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: ctx.redaction.restore_value(call.arguments.clone()),
        };
        &restored
    };
    // Before any policy is consulted or any prompt is opened: a call whose
    // arguments cannot be read as what the tool asked for is not a call the agent
    // meant to make, and running it with an empty set of arguments would be a
    // guess at someone's intent (MI-1).
    //
    // `LENIENT_READERS` opts a tool out of the shape check, not of the read: its
    // own reader accepts more than the description claims, on purpose.
    let shape = if LENIENT_READERS.contains(&call.name.as_str()) {
        None
    } else {
        ctx.schemas.get(&call.name)
    };
    let args = match call.arguments_for(shape) {
        Ok(args) => args,
        Err(reason) => {
            return err(format!("{} was not carried out: {reason}", call.name));
        }
    };
    // U-6: the reproduction gate is enforced here, at the one point every call
    // passes and before any prompt is opened, because what it refuses is not a
    // permission the user can grant — it is a write the fix has not earned yet.
    // While the gate waits for its failing reproduction, the only file the agent
    // may write is the reproduction itself.
    if tool_class(&call.name) == ToolClass::Edit {
        let target = arg_str(&args, "path")
            .and_then(|raw| workspace_path(root, raw).ok())
            .map(|(_, display)| display)
            .unwrap_or_default();
        if let crate::reprogate::WriteVerdict::Refused(reason) = ctx.repro.check_write(&target) {
            return err(reason);
        }
    }
    if let Some(headless) = &ctx.headless_policy {
        match headless.decide(root, &call.name, &args) {
            Headless::Refused { reason } => {
                ctx.record_approval(&call.name, tool_class(&call.name), ApprovalAnswer::Denied);
                return format!("error: {reason}");
            }
            Headless::Allow => {}
        }
    }
    match classify(
        root,
        &call.name,
        &args,
        ctx.mode,
        &ctx.granted(),
        ctx.tainted(),
    ) {
        // Refused without asking: the path is outside what the agent may
        // touch in any mode, so a prompt would only invite a mistake.
        Permission::Deny => FORBIDDEN_RESULT.to_string(),
        Permission::Allow => run_and_checkpoint(rt, root, call, ctx, mcp).await,
        Permission::Ask => {
            let class = tool_class(&call.name);
            // The prompt is rebuilt each time round, so what the person is
            // reading is what the file holds now. `Approved` is consent to the
            // change that was shown: if the file moved while the prompt was
            // open, the approval is not spent, it is asked again.
            let mut reshown = 0usize;
            loop {
                // One pass over the files for both halves, so the bytes the
                // person is shown and the bytes the approval is later checked
                // against are read together.
                let shown_all = approval_shown(root, call);
                let request = ApprovalRequest {
                    tool: call.name.clone(),
                    class,
                    summary: approval_summary(call),
                    preview: shown_all.preview,
                    draft: shown_all.draft,
                };
                let (responder, answer) = oneshot::channel();
                // Held out of the request, which moves into the prompt: the bytes
                // are only wanted once the person has answered, and the refusal is
                // drafted from the line the overlay was showing.
                let shown = request.draft.clone();
                let refusal_event = denied_event(&request);
                if ctx.prompts.send((request, responder)).is_err() {
                    // Nothing is listening — no TUI attached. The strictest
                    // possible answer is the only honest one, and it is an answer:
                    // the run went ahead having been refused, so the ledger says so.
                    ctx.record_approval(&call.name, class, ApprovalAnswer::Denied);
                    return UNATTENDED_RESULT.to_string();
                }
                let decision = match answer.await {
                    Ok(answer) => {
                        // A fetch is approved one address at a time. An "always allow"
                        // answer on one page is not a standing permission to reach the
                        // next, so this is recorded as the weaker thing it now is —
                        // otherwise the run's own history would claim a consent the
                        // gate refuses to honour, and `xencode runs` would show it.
                        let answer = if class == ToolClass::Network {
                            ApprovalAnswer::Approved
                        } else {
                            answer
                        };
                        ctx.record_approval(&call.name, class, answer);
                        if answer == ApprovalAnswer::Denied {
                            draft_denied_call(root, refusal_event);
                        }
                        answer
                    }
                    // A dropped responder means the prompt vanished with the app.
                    Err(_) => {
                        ctx.record_approval(&call.name, class, ApprovalAnswer::Denied);
                        ApprovalAnswer::Denied
                    }
                };
                match decision {
                    ApprovalAnswer::Approved => {
                        let moved = shown.stale_paths();
                        if moved.is_empty() {
                            return run_and_checkpoint(rt, root, call, ctx, mcp).await;
                        }
                        // The change that was agreed to is gone: writing now would land
                        // a different one. Nothing has been touched, and the person is
                        // asked again about the bytes that are actually there — until
                        // the file proves it cannot be held still.
                        if reshown >= MAX_DRAFT_REVIEWS {
                            return err(format!(
                                "not written: {} changed while the approval prompt was \
                                 open, and kept changing on every re-review. The change \
                                 that was approved is not the change that would land — \
                                 re-issue the call against the file as it stands now.",
                                moved.join(", ")
                            ));
                        }
                        reshown += 1;
                    }
                    // "Always allow everything like this" is the person waiving the
                    // per-change review, so there is no review left to invalidate: the
                    // grant covers the write however the file has moved.
                    ApprovalAnswer::ApprovedForSession => {
                        ctx.grant(class);
                        return run_and_checkpoint(rt, root, call, ctx, mcp).await;
                    }
                    ApprovalAnswer::Denied => return DENIED_RESULT.to_string(),
                }
            }
        }
    }
}

/// What a refused call is recorded as: the tool, the class of thing it would have
/// done, and the one line the overlay was showing (`/lesson` prints that line as
/// the evidence). A secret in the argument is scrubbed the same way the tool
/// result scrubs it, because the draft file is read by a person and indexed by
/// the project memory.
fn denied_event(request: &ApprovalRequest) -> xencode_context_rs::Evidence {
    xencode_context_rs::denied_call(
        &request.tool,
        request.class.overlay_label(),
        &request.summary,
    )
}

/// A person answering `n` is a decision about this repository, so it leaves the
/// same kind of draft a rewind does (`EV-7`'s gate, `QM-6`'s trigger). The reason
/// stays blank, because they did not give one and inventing it is the drift the
/// blank exists to prevent. A draft that cannot be written is not worth failing a
/// tool over — the refusal has already happened, and the call is not run either way.
fn draft_denied_call(root: &Path, event: xencode_context_rs::Evidence) {
    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
    let _ = xencode_context_rs::draft_lesson(&event, &xencode);
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

    /// The files one turn wrote, in the order they were snapshotted. Used to
    /// hand the turn's paths to the git checkpoint once the turn is over.
    pub fn group_paths(&self, turn: usize) -> Vec<PathBuf> {
        self.lock()
            .get(turn)
            .map(|group| group.iter().map(|one| one.full.clone()).collect())
            .unwrap_or_default()
    }

    /// The newest turn that wrote any file: the group `rewind(1)` would undo.
    /// A ByteBot task's undo (BT-3) checks this is its own turn, because two
    /// turns can write the same files and a file list cannot tell them apart.
    pub fn latest_write_turn(&self) -> Option<usize> {
        self.lock().iter().rposition(|group| !group.is_empty())
    }

    /// The files a `rewind(back)` would touch, without touching anything.
    /// `/rewind` asks git about these before restoring, so that a hand edit
    /// made after the agent's write is not silently overwritten.
    pub fn pending_paths(&self, back: usize) -> Vec<PathBuf> {
        let groups = self.lock();
        let mut out = Vec::new();
        let mut seen = 0;
        for group in groups.iter().rev() {
            if group.is_empty() {
                continue;
            }
            seen += 1;
            if seen > back {
                break;
            }
            for checkpoint in group {
                if !out.contains(&checkpoint.full) {
                    out.push(checkpoint.full.clone());
                }
            }
        }
        out
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

/// Consumers handed back to the model per `what_breaks` call. The same budget
/// the advice report uses, because both are read as one tool result.
const MODEL_IMPACT_CAP: usize = 40;

/// The answer, in the shape a model reads: the target, what it declares, then
/// the consumers grouped by how far back they are, each with the path the index
/// resolved. The basis line is not decoration — an edge here is a module path
/// that resolves, and a list of files that link to the target is a weaker claim
/// than a list of files that call what is being edited.
fn render_impact(report: &xencode_context_rs::ImpactReport) -> String {
    let mut lines: Vec<String> = vec![format!("what links to {}", report.target)];
    if !report.declared.is_empty() {
        let surface = if report.declared_more > 0 {
            format!(
                "{}, +{} more",
                report.declared.join(", "),
                report.declared_more
            )
        } else {
            report.declared.join(", ")
        };
        lines.push(String::new());
        lines.push(format!("It declares: {surface}"));
    }
    lines.push(String::new());
    if report.files.is_empty() {
        lines.push(if report.tier == xencode_context_rs::ImpactTier::Semantic {
            "Nothing outside this file refers to a symbol it defines.".to_string()
        } else {
            "Nothing in the index links to it: no file writes a `use` path to it, declares \
             it as a module, or implements a trait it defines."
                .to_string()
        });
    } else {
        let mut shown_hop = 0;
        for file in report.files.iter().take(MODEL_IMPACT_CAP) {
            if file.hops != shown_hop {
                shown_hop = file.hops;
                lines.push(if file.hops == 1 {
                    "Links to it directly:".to_string()
                } else {
                    format!("Reached through those, {} hops back:", file.hops)
                });
            }
            let semantic = report.tier == xencode_context_rs::ImpactTier::Semantic;
            let mut line = if semantic {
                format!("  {}  refers to {}", file.file, file.via.join(", "))
            } else {
                format!("  {}  via {}", file.file, file.via.join(", "))
            };
            // The semantic tier lists only files that refer to the symbol on the
            // first hop, so there is nothing to mark.
            if report.symbol.is_some() && !semantic {
                line.push_str(if file.uses_symbol {
                    " — its own `use` names it"
                } else {
                    " — does not name it"
                });
            }
            lines.push(line);
        }
        if report.files.len() > MODEL_IMPACT_CAP {
            lines.push(format!(
                "… +{} more, further back",
                report.files.len() - MODEL_IMPACT_CAP
            ));
        }
    }
    lines.push(String::new());
    lines.push(report.basis());
    if let (Some(symbol), xencode_context_rs::ImpactTier::Names) = (&report.symbol, report.tier) {
        lines.push(format!(
            "\"names it\" means `{symbol}` appears as a whole path segment in that file's own \
             `use` statements. A file that reaches this one through `mod` or `impl` has no `use` \
             to name it in, so it is reported as not naming it rather than as unrelated."
        ));
    }
    lines.join("\n")
}

fn tool_what_breaks(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(raw) = arg_str(args, "path") else {
        return err("what_breaks needs a string \"path\"");
    };
    let symbol = arg_str(args, "symbol").filter(|s| !s.is_empty());
    // The index only ever read Rust, so a file of another language has no
    // consumers recorded for it — say that, rather than reporting an empty list.
    if let Some(refusal) = xencode_context_rs::semantic_tier_refusal(
        "The dependency index",
        raw,
        xencode_context_rs::detect_language(Path::new(raw)),
    ) {
        return err(refusal);
    }
    // A fresh semantic index answers first (LSP-2). It is never built here —
    // that takes minutes — and a missing or stale one is named, so the reader
    // knows the answer below is the weaker, name-level one.
    let mut fallback_note = None;
    if let Ok(workspace) = xencode_context_rs::verify::manifest_dir(root) {
        match xencode_context_rs::scip_index::semantic_impact_for(
            &workspace,
            raw,
            symbol,
            xencode_context_rs::IMPACT_MAX_HOPS,
        ) {
            Ok(Ok(report)) => return render_impact(&report),
            Ok(Err(_)) => {
                fallback_note = Some("The semantic index does not hold this file.".to_string())
            }
            Err(why) => fallback_note = Some(format!("Not from the semantic index: {why}.")),
        }
    }
    match xencode_context_rs::impact_from_snapshot(root, raw, symbol) {
        Ok(report) => {
            let mut out = render_impact(&report);
            if let Some(note) = fallback_note {
                out.push_str("\n\n");
                out.push_str(&note);
            }
            out
        }
        Err(e) => err(e.to_string()),
    }
}

/// Longest one `debug_test` session may take, build included. Separate from
/// the command timeout, which is sized for one command and not for a build.
const DEBUG_TIMEOUT_SECS: u64 = 300;

/// `debug_test` (`CI-7`): run one test under a real debugger until a line, and
/// report where it stopped and what the frame held. One call is one whole
/// session; nothing is left running afterwards.
async fn tool_debug_test(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(test) = arg_str(args, "test").filter(|t| !t.trim().is_empty()) else {
        return err("debug_test needs a string \"test\": the test's full name, as `cargo test -- --list` prints it");
    };
    let Some(file) = arg_str(args, "file").filter(|f| !f.trim().is_empty()) else {
        return err("debug_test needs a string \"file\": the source file to stop in");
    };
    let Some(line) = args.get("line").and_then(|v| v.as_u64()).filter(|l| *l > 0) else {
        return err("debug_test needs a positive \"line\" to stop at");
    };
    let workspace = match xencode_context_rs::verify::manifest_dir(root) {
        Ok(ws) => ws,
        Err(e) => return err(format!("debug_test needs a Cargo workspace: {e}")),
    };
    // The file is given root-relative like every other tool's path; the
    // debugger is handed it absolute, so the workspace's place does not matter.
    let (full, display) = match workspace_path(root, file) {
        Ok(ok) => ok,
        Err(reason) => return reason,
    };
    let request = xencode_analysis_rs::dap::DebugRequest {
        workspace,
        package: arg_str(args, "package").map(str::to_string),
        test: test.to_string(),
        file: full.to_string_lossy().into_owned(),
        line: line as u32,
        timeout: std::time::Duration::from_secs(DEBUG_TIMEOUT_SECS),
    };
    let outcome =
        match tokio::task::spawn_blocking(move || xencode_analysis_rs::dap::debug_test(&request))
            .await
        {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(e)) => return err(format!("debug_test: {e}")),
            Err(join) => return err(format!("debug_test stopped unexpectedly: {join}")),
        };
    let mut lines = vec![format!("debugger: {}", outcome.adapter)];
    match &outcome.stopped_at {
        Some(at) => {
            lines.push(format!(
                "stopped ({}) in {at}",
                outcome.reason.as_deref().unwrap_or("unknown reason")
            ));
            if outcome.variables.is_empty() {
                lines.push("the frame showed no locals".to_string());
            }
            for var in &outcome.variables {
                let ty = var
                    .ty
                    .as_ref()
                    .map(|t| format!(": {t}"))
                    .unwrap_or_default();
                lines.push(format!("  {}{ty} = {}", var.name, var.value));
            }
        }
        None => lines.push(format!(
            "the test ran to the end without reaching {display}:{line}{} — the line is \
             not on the path this test takes, or holds no code",
            outcome
                .exit_code
                .map(|c| format!(" (exit code {c})"))
                .unwrap_or_default()
        )),
    }
    lines.push("The debugger and the test process have both been ended.".to_string());
    lines.join("\n")
}

/// Most references or callers listed to the model in one answer.
const MODEL_REFS_CAP: usize = 60;

/// `find_refs` and `callers` (`LSP-1`): every place one Rust item is written, or
/// the functions that use it, from rust-analyzer's semantic index. Read-only,
/// and never builds the index — that takes minutes — so a missing or stale one
/// is reported with how to build it.
fn tool_find_refs(
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    callers: bool,
) -> String {
    use xencode_context_rs::scip_index::{self, Pick};
    let tool = if callers { "callers" } else { "find_refs" };
    let Some(symbol) = arg_str(args, "symbol").filter(|s| !s.trim().is_empty()) else {
        return err(format!("{tool} needs a string \"symbol\""));
    };
    let in_file = arg_str(args, "in").filter(|s| !s.trim().is_empty());
    let workspace = match xencode_context_rs::verify::manifest_dir(root) {
        Ok(ws) => ws,
        Err(e) => return err(format!("{tool} reads a Cargo workspace's index: {e}")),
    };
    // Shown paths are root-relative, like every other tool's.
    let prefix = match workspace.strip_prefix(root) {
        Ok(rel) if !rel.as_os_str().is_empty() => {
            format!("{}/", rel.to_string_lossy().replace('\\', "/"))
        }
        _ => String::new(),
    };
    let in_workspace = in_file.map(|f| {
        f.trim()
            .replace('\\', "/")
            .trim_start_matches(prefix.as_str())
            .to_string()
    });
    let found = match scip_index::symbols_named(&workspace, symbol) {
        Ok(found) => found,
        Err(why) => {
            return err(format!(
                "{tool} answers only from rust-analyzer's semantic index, and {why}. \
                 search_files finds the text instead, without knowing which item it is"
            ))
        }
    };
    let item = match scip_index::pick(found, symbol, in_workspace.as_deref()) {
        Pick::One(item) => item,
        Pick::Nothing(why) => return err(why),
        Pick::Several(items) => {
            let list = items
                .iter()
                .map(|n| {
                    let files: Vec<String> = n
                        .defined_in
                        .iter()
                        .map(|f| format!("{prefix}{f}"))
                        .collect();
                    format!("a {} in {}", n.kind, files.join(" and "))
                })
                .collect::<Vec<_>>()
                .join("; ");
            return err(format!(
                "`{symbol}` names {} different items — {list}. Pass \"in\" with the file \
                 that declares the one you mean",
                items.len()
            ));
        }
    };
    let refs = scip_index::references(&workspace, &item, symbol);
    let declared: Vec<String> = item
        .defined_in
        .iter()
        .map(|f| format!("{prefix}{f}"))
        .collect();
    let mut lines = Vec::new();
    if callers {
        let mut by_caller: std::collections::BTreeMap<(String, String), usize> =
            std::collections::BTreeMap::new();
        let mut outside = 0usize;
        for r in refs.iter().filter(|r| !r.definition) {
            match &r.caller {
                Some(caller) => {
                    *by_caller
                        .entry((format!("{prefix}{}", r.file), caller.clone()))
                        .or_default() += 1
                }
                None => outside += 1,
            }
        }
        lines.push(format!(
            "functions that use `{symbol}` ({} in {}): {}",
            item.kind,
            declared.join(", "),
            by_caller.len()
        ));
        for ((file, caller), count) in by_caller.iter().take(MODEL_REFS_CAP) {
            lines.push(format!("  {caller}  in {file} ({count} use(s))"));
        }
        if by_caller.len() > MODEL_REFS_CAP {
            lines.push(format!("  … +{} more", by_caller.len() - MODEL_REFS_CAP));
        }
        if outside > 0 {
            lines.push(format!(
                "{outside} more use(s) sit outside any function: `use` lines, types, \
                 constants. find_refs lists them."
            ));
        }
    } else {
        lines.push(format!(
            "`{symbol}` ({} in {}): {} place(s)",
            item.kind,
            declared.join(", "),
            refs.len()
        ));
        for r in refs.iter().take(MODEL_REFS_CAP) {
            let role = if r.definition {
                " — declared here".to_string()
            } else {
                r.caller
                    .as_ref()
                    .map(|c| format!(" — in {c}"))
                    .unwrap_or_default()
            };
            lines.push(format!("  {prefix}{}:{}{role}  {}", r.file, r.line, r.text));
        }
        if refs.len() > MODEL_REFS_CAP {
            lines.push(format!("  … +{} more", refs.len() - MODEL_REFS_CAP));
        }
    }
    lines.push(String::new());
    lines.push(
        "From rust-analyzer's semantic index: each place resolves to this one item, \
         through re-exports and other crates; a field, local or other item with the \
         same name is not included. The function each use sits in is read from the \
         file's parse tree."
            .to_string(),
    );
    lines.join("\n")
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
mod tests;
