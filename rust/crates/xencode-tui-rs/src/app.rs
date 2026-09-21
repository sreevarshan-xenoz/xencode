use std::collections::{HashMap, HashSet};
use std::io;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind, MouseButton, MouseEventKind};
use ratatui::{backend::Backend, Terminal};
use tokio::sync::mpsc;
use tui_textarea::{CursorMove, TextArea};

use xencode_config_rs::XencodeConfig;
use xencode_context_rs::{init_project, DocError, DocText, HardwareProfile};
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::{
    current_timestamp, find_llama_server, resolve_gguf_model, start_llama_server, HealthStatus,
    LlamaCppClient, LlamaCppOptions, LlamaCppTimings, LlamaServerProcess, OllamaClient,
};
use xencode_providers_rs::{
    ChatMessage, ContentPart, ImageUrlPart, MessageContent, ProviderManager,
};

pub use crate::focus::{navigate_feature, FocusArea, InputMode, FEATURE_LIST};
pub use crate::theme::ThemeColors;
use crate::ui;

/// System block injected as tier 1 when previewing `/ctx` context assembly.
const CTX_SYSTEM: &str = xencode_context_rs::AGENT_SYSTEM_PROMPT;

/// Hardware profile the live chat path budgets against. Must stay in sync
/// with the llama.cpp `--ctx-size` the auto-start uses for this profile.
const CTX_PROFILE: HardwareProfile = HardwareProfile::Balanced;

/// Represents a message in the UI chat list
pub struct UiMessage {
    pub role: String,
    pub content: String,
}

/// Read `git status` into a map keyed exactly as `file_tree` is.
///
/// `file_tree` comes from `scan_workspace`, which stores each entry's path
/// relative to the root with no prefix (`src/main.rs`) — the same shape git
/// reports — so the path is used verbatim as the key.
///
/// `-z` rather than plain `--porcelain`: it emits paths NUL-terminated and
/// unquoted, so a name with non-ASCII or special characters arrives verbatim
/// instead of quoted and octal-escaped, and a rename's two paths are separate
/// fields instead of being joined by a literal " -> ".
fn git_status_map() -> HashMap<String, String> {
    let Ok(output) = Command::new("git")
        .args(["status", "--porcelain", "-z"])
        .output()
    else {
        return HashMap::new();
    };
    parse_porcelain_z(&output.stdout)
}

/// Parse the output of `git status --porcelain -z`.
///
/// Split out from [`git_status_map`] so it can be tested without a repository.
fn parse_porcelain_z(stdout: &[u8]) -> HashMap<String, String> {
    let mut status = HashMap::new();
    let mut fields = stdout.split(|&byte| byte == 0);
    while let Some(entry) = fields.next() {
        // Each entry is `XY <path>`; anything shorter is the trailing empty
        // field left by the final NUL.
        if entry.len() < 4 {
            continue;
        }
        let (code, path) = entry.split_at(3);

        // A rename or copy is followed by its original path as a separate
        // field. Consume it, or every subsequent entry is misread. The new
        // path comes first, and that is the one on disk, so it is the one to
        // key on.
        if matches!(code[0], b'R' | b'C') {
            fields.next();
        }

        status.insert(
            String::from_utf8_lossy(path).into_owned(),
            String::from_utf8_lossy(&code[..2]).trim().to_string(),
        );
    }
    status
}

/// Max recalled prompts kept per session.
const INPUT_HISTORY_LIMIT: usize = 200;

/// Slash commands intercepted by `submit_message`, in handler order.
pub const SLASH_COMMANDS: &[&str] = &[
    "/init", "/ctx", "/advise", "/bytebot", "/spawn", "/plan", "/rewind", "/mcp",
];

/// Complete a partially typed command token against `SLASH_COMMANDS`.
/// Returns the longest common prefix when it extends the token (pure —
/// unit-tested); None when ambiguous or already complete.
pub fn complete_slash_token(token: &str) -> Option<String> {
    if token.len() < 2 || !token.starts_with('/') {
        return None;
    }
    let matches: Vec<&'static str> = SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|c| c.starts_with(token))
        .collect();
    if matches.is_empty() {
        return None;
    }
    let mut lcp = matches[0].to_string();
    for m in &matches[1..] {
        let keep = lcp
            .chars()
            .zip(m.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a)
            .collect::<String>();
        lcp = keep;
    }
    (lcp.len() > token.len()).then_some(lcp)
}

/// Who is watching a tool-loop run (I2-04). The chat turn and ByteBot execute
/// the same rounds through the same permission gate — only where the events
/// go differs, so neither can drift from the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopSink {
    /// Stream deltas and `⚙` transcript lines, as the chat loop always has.
    Chat,
    /// Report to the ByteBot panel: calls become step rows, the model's text
    /// for a round becomes one log line.
    ByteBot,
    /// Report to the `/spawn` registry: each delegated subagent (I3-03) has
    /// its own id, and a finished run posts its report into the chat.
    Spawn(u64),
}

/// ByteBot panel events. `call:<summary>` opens a step row, `done:<outcome>`
/// retires the row still running, `log:<text>` appends a log line, `err:<text>`
/// reports a real failure. Progress is derived from the step rows on the app
/// side — nothing here invents it.
const BYTEBOT_PREFIX: &str = "[BYTEBOT]";

/// `/spawn` subagent events, one prefix per run id:
/// `[SPAWN]<id>:call:<summary>`, `:done:<outcome>`, `:log:<text>`,
/// `:err:<text>` (real failure) and finally `:finish:<final text>`. The app
/// side derives progress from the step rows and posts the report to the chat.
const SPAWN_PREFIX: &str = "[SPAWN]";

/// Findings the security panel streams before it stops listing them. The
/// totals it reports stay true — the cap only limits lines on screen.
const FINDINGS_CAP: usize = 200;

/// How long the profiler watches this process to turn two `/proc` reads into a
/// CPU rate. Long enough to be measurable, short enough to not feel like a
/// frozen UI.
const PROFILER_SAMPLE_MS: u64 = 250;

/// Persisted metric rows the profiler lists, newest first.
const PROFILER_METRIC_ROWS: usize = 6;

/// How a spawn's task is framed for the model. The worktree is the whole
/// deal: everything it touches is inside it, and it must say that plainly.
const SPAWN_BRIEF: &str = "Delegated task in your own git worktree — you are isolated \
                           from the main checkout, so read before you edit and test what \
                           you change; your changes land in this worktree only. Post your \
                           steps with update_plan as you go. Stop when it is done or when \
                           you are blocked, and say which; never report an outcome you \
                           did not observe.\n\nTask: ";

/// One `/spawn <task> [#branch]` subagent (I3-03). Kept on the app so
/// `/spawn status` is honest without a provider: the record mutates only from
/// the events `agent_rounds` reports, exactly like the ByteBot panel.
pub struct SpawnRecord {
    pub id: u64,
    pub branch: String,
    pub path: std::path::PathBuf,
    pub task: String,
    pub running: bool,
    pub failed: bool,
    pub steps: Vec<(String, String)>,
}

impl SpawnRecord {
    fn finished_line(&self) -> String {
        let done = self.steps.iter().filter(|(_, s)| s == "done").count();
        format!("{done}/{} call(s) completed", self.steps.len())
    }
}

/// How a delegated run is framed for the model. Deliberately short: the tools
/// themselves are taught by `TOOL_HINT`, which rides on the system turn.
const BYTEBOT_BRIEF: &str = "Delegated task — work it end to end with the tools, \
                             reading before you edit and testing what you change. \
                             Post your steps with update_plan as you go. Stop when it \
                             is done or when you are blocked, and say which; never \
                             report an outcome you did not observe.\n\nTask: ";

/// What the ByteBot progress bar means: the share of calls made so far that
/// came back. It can move backwards when the model makes another call — which
/// is honest, unlike a bar that hits 100% because a script promised six steps.
fn bytebot_progress(steps: &[(String, String)]) -> f64 {
    if steps.is_empty() {
        return 0.0;
    }
    let done = steps.iter().filter(|(_, s)| s != "running").count();
    done as f64 / steps.len() as f64
}

/// Everything the shared tool loop needs beyond the conversation itself.
/// Owned, because the loop runs on its own task.
struct AgentRun {
    sink: LoopSink,
    model: String,
    context_messages: Vec<ChatMessage>,
    approval: crate::agent_tools::ApprovalCtx,
    task_runtime: crate::agent_tools::TaskRuntime,
    tool_root: std::path::PathBuf,
    /// Offered until the final round, which is tool-less so a run always ends
    /// with a text answer.
    max_rounds: usize,
    /// Alternate models tried in order when the primary fails before emitting
    /// any output (I4-01). Set from `agent_fallback_models` config.
    fallback_models: Vec<String>,
    ollama_url: String,
    llama_cpp_url: String,
    timeout: u64,
    openrouter_key: Option<String>,
    qwen_key: Option<String>,
    gemini_key: Option<String>,
    llama_opts: LlamaCppOptions,
}

pub struct App<'a> {
    pub focus: FocusArea,
    /// Chat input box (multiline-capable; Enter submits, Alt+Enter/Ctrl+J
    /// insert newlines).
    pub chat_input: TextArea<'static>,
    pub input_mode: InputMode,
    pub messages: Vec<UiMessage>,
    pub chat_scroll: u16,
    /// Sent prompts, oldest first (chat input recall, Alt+Up/Down).
    pub input_history: Vec<String>,
    history_index: Option<usize>,
    history_draft: String,
    pub file_tree: Vec<String>,
    pub selected_file: usize,
    pub attached_files: HashSet<String>,
    pub opened_file: Option<String>,
    pub editor: TextArea<'a>,
    pub editor_dirty: bool,
    pub git_status: HashMap<String, String>,
    pub git_branch: String,
    pub available_models: Vec<String>,
    pub selected_model: usize,
    pub is_generating: bool,
    pub is_reviewing: bool,
    pub code_review_output: String,
    pub review_dash: crate::review::ReviewDashboard,
    /// Task panel (D2-01) cursor state. The data itself lives in
    /// `task_runtime`; these are just the view's selection/scroll.
    pub tasks_selected: usize,
    pub tasks_detail: bool,
    pub tasks_scroll: usize,
    /// WorktreePanel state (D3-02): `worktree_dirty[i]` flags whether
    /// `worktrees[i]` has uncommitted changes. Git calls are synchronous,
    /// matching `refresh_git` (Ctrl+G) precedent.
    pub worktrees: Vec<xencode_context_rs::WorktreeInfo>,
    pub worktree_dirty: Vec<bool>,
    pub worktree_selected: usize,
    pub worktree_prompt: crate::focus::WorktreePrompt,
    pub worktree_path_buf: String,
    pub worktree_branch_buf: String,
    pub worktree_status: String,
    /// AdvisePanel (F2-01): insights computed from the live `.xencode`
    /// snapshot on open/`r`; the detail view scrolls with `advise_scroll`.
    pub advise_items: Vec<xencode_context_rs::Advice>,
    pub advise_selected: usize,
    pub advise_detail: bool,
    pub advise_scroll: usize,
    pub advise_status: String,
    pub commit_message: String,
    pub commit_cursor: usize,
    pub spinner_tick: usize,
    pub theme: ThemeColors,
    pub config: XencodeConfig,
    /// Product builds persist config edits; `for_tests()` turns that off so a
    /// keystroke in a test never rewrites the developer's `config.json`.
    pub(crate) persist_config: bool,
    pub show_terminal: bool,
    /// Last focus that pointed at a body pane (explorer/editor/chat). Zen
    /// layout uses it as its focus-follows target; overlay focus never
    /// touches it, so zen doesn't flicker when a popup closes.
    pub last_body_focus: FocusArea,
    /// The body geometry `draw_body` rendered last frame. The Tab focus-ring
    /// reads it so it can only Tab to panes the user can actually see.
    pub last_layout: crate::layout::BodyLayout,
    /// Tool classes the user answered "always allow" for this session
    /// (I1-03 approvals). Session-only: never persisted. Shared with the
    /// spawned tool loops so a grant made mid-turn holds for the next one.
    pub agent_grants: Arc<std::sync::Mutex<Vec<crate::agent_tools::ToolClass>>>,
    /// Byte-for-byte snapshots of what the agent changed, grouped per chat
    /// turn (I2-01). `/rewind` puts them back; quitting drops them.
    pub checkpoints: Arc<crate::agent_tools::CheckpointStore>,
    /// The agent's current todo list (I2-03), written by its `update_plan`
    /// calls and rendered as a strip above the transcript. Session-only.
    pub agent_plan: crate::agent_tools::PlanHandle,
    /// `/plan` toggles this: pinned shows every item, unpinned the first few.
    pub plan_pinned: bool,
    /// MCP servers started for this session (I3-01) and the tools they offer.
    /// Empty until `/mcp` starts something.
    pub mcp: Arc<crate::mcp::McpHub>,
    /// Pending approval prompts from the agent tool loop, in arrival order.
    /// The overlay shows the front; answering pops and resolves the oneshot
    /// the tool task is awaiting.
    pub approval_queue: std::collections::VecDeque<(
        crate::agent_tools::ApprovalRequest,
        tokio::sync::oneshot::Sender<crate::agent_tools::ApprovalAnswer>,
    )>,
    pub approval_scroll: usize,
    /// Sender the spawned tool loops use to raise approval prompts.
    pub approval_tx: mpsc::UnboundedSender<(
        crate::agent_tools::ApprovalRequest,
        tokio::sync::oneshot::Sender<crate::agent_tools::ApprovalAnswer>,
    )>,
    /// Receive end, taken once by `run_app` and drained each frame.
    pub approval_rx: Option<
        mpsc::UnboundedReceiver<(
            crate::agent_tools::ApprovalRequest,
            tokio::sync::oneshot::Sender<crate::agent_tools::ApprovalAnswer>,
        )>,
    >,
    pub memory: ConversationMemory,
    pub feature_nav_selected: usize,

    // Performance & health tracking
    pub session_start_time: f64,
    pub ollama_health_entries: HashMap<String, (String, f64, Option<String>)>,
    pub last_health_check: f64,
    pub health_check_in_progress: bool,
    /// Handle to a llama-server that the TUI auto-started this session (if any).
    pub llama_process: Option<Arc<std::sync::Mutex<Option<LlamaServerProcess>>>>,
    /// Set when xencode exits so a still-loading auto-start aborts instead of
    /// orphaning a server behind the app.
    pub llama_cancel: Arc<AtomicBool>,
    /// Shared background-task registry driven by the D1-02 tool loop (and the
    /// D2 task panel/CLI later).
    pub task_runtime: crate::agent_tools::TaskRuntime,
    pub total_llm_calls: u64,
    pub average_latency: f64,

    // ByteBot state
    pub bytebot_command: String,
    pub bytebot_cursor: usize,
    pub bytebot_steps: Vec<(String, String)>, // (step_name, status)
    pub bytebot_progress: f64,
    pub bytebot_running: bool,
    pub bytebot_log: Vec<String>,
    pub bytebot_history: Vec<String>, // previously executed commands
    /// `/spawn` subagent registry (I3-03): every delegated run is isolated in
    /// its own git worktree, and the record below is what `/spawn status`
    /// reads. Grows over the session; ids never collide.
    pub spawns: Vec<SpawnRecord>,
    pub spawn_next_id: u64,

    // Project context (M0) state
    /// Whether the "run /init" hint was already shown this session, so a
    /// missing project index nudges once instead of on every message.
    pub context_hint_shown: bool,
    pub init_running: bool,
    pub init_progress: f64,
    pub init_steps: Vec<(String, String)>, // (step_name, status)
    pub init_log: Vec<String>,
    pub init_visible: bool,
    pub help_visible: bool,
    pub help_scroll: u16,
    /// Transient notifications (file-watch warnings etc.), see `toast` module.
    pub toasts: Vec<crate::toast::Toast>,
    pub init_cancel: Arc<AtomicBool>,

    // Collaboration Hub state
    pub collab_session_active: bool,
    pub collab_session_id: String,
    pub collab_members: Vec<(String, String, String)>, // (name, role, connection)
    pub collab_sync_status: String,                    // "disconnected", "connecting", "connected"
    pub collab_last_sync: f64,
    pub collab_activity_log: Vec<String>,
    pub collab_server_url: String,
    pub collab_username: String,
    /// The live client task, if any. Aborted when the hub disconnects.
    pub collab_worker: Option<tokio::task::JoinHandle<()>>,
    /// The server's most recent `error` frame, for the hub to surface.
    pub collab_error: String,
    /// Hub form mode: while set, typed characters edit `collab_field`.
    pub collab_editing: bool,
    pub collab_field: crate::focus::CollabField,

    // Voice Interface state
    pub voice_active: bool,
    /// "idle", "listening" or "processing" — nothing else, because nothing
    /// else happens: there is no text-to-speech in this product (J-07).
    pub voice_status: String,
    /// RMS of the last PCM chunk the recorder produced, after the display gain.
    pub voice_level: f64,
    pub voice_peak: f64,
    pub voice_pcm_bytes: usize,
    /// Speech text from a transcriber, and nothing else.
    pub voice_transcript: Vec<String>,
    /// What the session is about: the clip path, or why there is no text.
    pub voice_note: String,
    pub voice_clip: Option<std::path::PathBuf>,
    /// Which recorder answered this session, by file name.
    pub voice_recorder: String,
    pub voice_muted: bool,
    pub voice_busy: bool,
    /// Set by the app to end the capture loop in the blocking task.
    pub voice_stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Read and written across the capture task so `m` takes effect mid-clip.
    pub voice_mute_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub voice_root: std::path::PathBuf,

    // Terminal Assistant state
    /// A suggestion request is with the provider right now.
    pub term_asst_busy: bool,
    pub term_asst_query: String,
    /// The query field owns the keyboard: letters type instead of selecting.
    pub term_asst_typing: bool,
    /// (command, risk label, why the model suggested it)
    pub term_asst_suggestions: Vec<(String, String, String)>,
    pub term_asst_selected: usize,
    pub term_asst_output: String,
    pub term_asst_history: Vec<(String, String, String)>, // (command, risk, outcome)
    /// "All", "safe" or "destructive" — matches the labels `parse_term_suggestions`
    /// emits, so the filter can only ever show rows that exist.
    pub term_risk_filter: String,

    // Security Auditor state
    pub sec_scan_active: bool,
    pub sec_scan_path: String,
    pub sec_scan_results: Vec<(String, String, String)>, // (severity, category, file)
    pub sec_scan_summary: (u32, u32, u32, u32),          // (critical, high, medium, low)
    pub sec_scan_progress: f64,
    pub sec_scan_log: Vec<String>,
    pub sec_filter_severity: String, // "All", "Critical", "High", "Medium", "Low"
    pub sec_sort_mode: String,       // "severity" or "category"

    // Performance Profiler state. Everything here is measured, not simulated:
    /// `None` means the number does not exist yet (no turn has run, `/proc`
    /// unreadable), which the panel renders as `n/a` rather than as zero.
    pub profiler_active: bool,
    pub profiler_running: bool,
    pub profiler_rows: Vec<(String, String, String)>, // (source, metric, value)
    pub profiler_notes: Vec<String>,
    pub profiler_gauge_cpu: Option<f64>, // % of one core, this process
    pub profiler_gauge_mem: Option<f64>, // resident set size, MB
    pub profiler_gauge_mem_total: Option<f64>, // system memory, MB (bar scale)
    pub profiler_gauge_latency: Option<f64>, // average turn latency, ms

    // Custom Models state
    /// The profiles as the panel holds them: read from `config.json` at
    /// startup, edited by `-`/`+` and `←`/`→`, and written back only when `s`
    /// succeeds. Nothing here is seeded.
    pub model_profiles: Vec<xencode_config_rs::ModelProfile>,
    pub models_selected: usize,
    /// Parameters moved since the last save, so the panel can say the list on
    /// disk and the list on screen are no longer the same.
    pub models_dirty: bool,
    /// The last thing the panel did — applied, saved, refused, or the
    /// provider's own words after a test request.
    pub models_status: String,
    /// A test request is with the provider right now.
    pub models_busy: bool,

    // Learning Mode state: lessons are files the project index says exist, so
    // the queue, the source on screen and the answer key all come from outside
    // this panel. See `start_learning`.
    pub learn_active: bool,
    /// The workspace the queue was built from, so `p`/`n` cannot drift onto a
    /// different checkout mid-session.
    pub learn_root: std::path::PathBuf,
    /// The queue: `(repo-relative path, what it declares)`, from `.xencode`.
    pub learn_lessons: Vec<(String, Vec<String>)>,
    /// Index into `learn_lessons` for the lesson on screen.
    pub learn_current_lesson: usize,
    pub learn_total_lessons: usize,
    /// The file this lesson is about — a path, not a invented title.
    pub learn_lesson_title: String,
    /// Facts read off the index: what the file declares, and what was sent.
    pub learn_content: Vec<String>,
    /// The file's own text, capped, and what the model was given.
    pub learn_code_example: String,
    /// The panel's own report: a refusal, a provider error, the model's why.
    pub learn_status: String,
    /// An explanation request is with the provider right now.
    pub learn_busy: bool,
    pub learn_quiz_active: bool,
    pub learn_quiz_question: String,
    pub learn_quiz_options: Vec<String>,
    pub learn_quiz_selected: usize,
    pub learn_quiz_answered: bool,
    pub learn_quiz_correct: bool,
    /// Which option the model called correct. `None` until it says so.
    pub learn_quiz_answer: Option<usize>,
    /// The model's own sentences about this file, kept apart from the facts the
    /// index produced so the panel can say which is which.
    pub learn_explain: Vec<String>,
    /// The model's reason for its answer key, shown once graded.
    pub learn_quiz_why: String,

    // Multi-Language state
    /// A walk or a translation request is in flight; the panel does one thing
    /// at a time.
    pub lang_busy: bool,
    /// The directory the last walk covered; empty means it never ran.
    pub lang_scan_path: String,
    /// (language, files, lines, share of the workspace's lines)
    pub lang_detection_results: Vec<(String, u64, u64, f64)>,
    /// What the walk could not answer: a failure, files skipped, files unread.
    pub lang_notes: Vec<String>,
    pub lang_translate_input: String,
    pub lang_translate_output: String,
    pub lang_translate_source: String,
    pub lang_translate_target: String,
    /// The last request failed, so the output line is an error and is drawn as
    /// one rather than in the colour of an answer.
    pub lang_translate_error: bool,
    /// Which field owns the keyboard; `None` means the keys are commands.
    pub lang_editing: Option<crate::focus::LangField>,

    // Panel scroll state
    pub provider_health_scroll: u16,
    pub security_scroll: u16,
    pub review_scroll: u16,

    // Settings interactive state
    pub settings_cursor: usize,
    pub settings_reset_active: bool,
    pub settings_url_editing: bool,
    pub settings_url_buffer: String,
    pub settings_url_cursor: usize,

    // llama.cpp live-control state (model load/unload + sampling)
    pub llamacpp_editing: bool,
    pub llamacpp_path_buffer: String,
    pub llamacpp_path_cursor: usize,
    pub llamacpp_action_msg: String,
    // llama.cpp sampling options (temperature, top-k, min-p, max-tokens)
    pub sampling_temp_editing: bool,
    pub sampling_temp_buffer: String,
    pub sampling_int_editing: bool,
    pub sampling_int_buffer: String,

    // Last llama.cpp generation timings (tok/s) reported by the server
    pub last_llamacpp_timings: Option<LlamaCppTimings>,

    /// Total prompt tokens of the last `/ctx` assembly preview — used to derive
    /// `cached_tokens` from llama.cpp `tokens_evaluated` (§13).
    pub last_ctx_total_tokens: u64,
    /// Retrieved files in the last assembly (recorded into metrics row).
    pub last_ctx_retrieved_files: u8,
}

/// First non-empty line of a process output, for one-line chat reporting.
/// Pure — unit-tested.
pub fn first_output_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("(no output)")
        .to_string()
}

/// Format the proactive warning for a watched path. Pure — unit-tested.
pub fn format_watch_warning(path: &str, kind: &str) -> String {
    match kind {
        "removed" => {
            format!("⚠ {path} was removed from disk — re-add or restore it before relying on it.")
        }
        "created" => format!("⚠ {path} was created on disk — it may affect your plan."),
        _ => format!(
            "⚠ {path} changed on disk — re-read it before relying on the version in context."
        ),
    }
}

/// Live snapshot hook (F1-02): rewrite the `.xencode` symbol/dep snapshot
/// after a modified/removed `.rs` file so the dependent list (and `/advise`)
/// see current imports. Returns whether the snapshot was rewritten. Errors
/// are swallowed deliberately — a stale snapshot still warns, a panic
/// on the UI thread would not.
pub fn live_refresh_snapshot(root: &std::path::Path, kind: &str, path: &str) -> bool {
    if !matches!(kind, "modified" | "removed") || !path.ends_with(".rs") {
        return false;
    }
    matches!(
        xencode_context_rs::refresh_rust_file(root, path),
        Ok(xencode_context_rs::RefreshOutcome::Updated(_))
    )
}

/// Decide whether a watched path deserves a proactive warning. Pure —
/// unit-tested.
///
/// `attached`/`opened`/`tracked` are the session's context sets;
/// `dependents` are the files transitively importing `path` (from the live
/// `.xencode` snapshot, kept current by [`live_refresh_snapshot`] and
/// otherwise by the last `/init`); `last_notice` is the newest visible warning
/// toast, used to suppress repeat warnings for a path the user was
/// already told about.
pub fn watch_warning_for(
    path: &str,
    kind: &str,
    attached: &HashSet<String>,
    opened: Option<&str>,
    tracked: &HashSet<String>,
    dependents: &[String],
    last_notice: Option<&str>,
) -> Option<String> {
    if !(attached.contains(path) || opened == Some(path) || tracked.contains(path)) {
        return None;
    }
    // Suppress only the exact warning already shown: a substring match
    // misfires on sibling paths (src/log.rs vs src/blog.rs) and on the
    // dependents list itself, and a kind change (modified → removed) must
    // re-warn because the head text differs.
    let head = format_watch_warning(path, kind);
    if last_notice.is_some_and(|m| m.contains(&head)) {
        return None;
    }
    let mut warning = head;
    if !dependents.is_empty() {
        const SHOW: usize = 5;
        let shown = dependents
            .iter()
            .take(SHOW)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let rest = dependents.len().saturating_sub(SHOW);
        let more = if rest > 0 {
            format!(" (+{rest} more)")
        } else {
            String::new()
        };
        warning.push_str(&format!("\n↳ Dependents to re-check: {shown}{more}"));
    }
    Some(warning)
}

/// Read an attached document off disk and parse it to text. Returns a short
/// human reason on failure for the skip note — same visibility contract as
/// images: never silent.
fn parse_attached_document(path: &str) -> Result<DocText, String> {
    use xencode_context_rs::{parse_document_bytes, MAX_DOC_BYTES};
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read file: {e}"))?;
    if bytes.len() > MAX_DOC_BYTES {
        return Err(format!(
            "exceeds the {} MiB document cap",
            MAX_DOC_BYTES / 1024 / 1024
        ));
    }
    parse_document_bytes(path, &bytes).map_err(|e| match e {
        DocError::UnknownType(_) => "not a supported document".to_string(),
        DocError::ParseError(_, kind, detail) => format!("{kind} parse failed: {detail}"),
        DocError::ReadError(_, detail) => format!("cannot read file: {detail}"),
        DocError::TooLarge(_, _) => "exceeds the document cap".to_string(),
    })
}

/// Render a parsed document as an attached-block entry: extracted text when
/// present, an explicit note when the document held nothing extractable
/// (scanned PDFs) or failed to parse. Pure — unit-tested.
fn doc_attach_block(path: &str, parsed: &Result<DocText, String>) -> String {
    match parsed {
        Ok(doc) if !doc.text.trim().is_empty() => {
            format!("<file path=\"{path}\">\n{}\n</file>\n\n", doc.text)
        }
        Ok(_) => format!(
            "<file path=\"{path}\">\n(document held no extractable text — scanned?)\n</file>\n\n"
        ),
        Err(reason) => {
            format!("<file path=\"{path}\">\n(document not parsed: {reason})\n</file>\n\n")
        }
    }
}

/// Append a text attachment to the attached block, or a visible skip note
/// when the file cannot be read as text (binary, deleted, permissions).
/// Pure over the file — unit-tested. Same never-silent contract as images
/// and documents.
fn append_text_attachment(block: &mut String, path: &str) {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            block.push_str(&format!("<file path=\"{path}\">\n{content}\n</file>\n\n"));
        }
        Err(e) => {
            block.push_str(&format!(
                "<file path=\"{path}\">\n(attachment not sent: {e})\n</file>\n\n"
            ));
        }
    }
}

/// Read an attached image off disk and encode it as a data URL for message
/// parts. Pure over the file — unit-tested. The `Err` reason is a short
/// human phrase for the `(image not sent: …)` note in the attached block,
/// so a skipped image is always visible, never silent.
fn encode_attached_image(path: &str) -> Result<String, String> {
    use xencode_analysis_rs::{inspect_bytes, to_data_url, ImageError};
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read file: {e}"))?;
    let meta = inspect_bytes(path, &bytes).map_err(|e| match e {
        ImageError::TooLarge(_, cap) => {
            format!("exceeds the {} MiB image cap", cap / 1024 / 1024)
        }
        ImageError::UnknownFormat(_) => "not a recognized image".to_string(),
        ImageError::ReadError(_, detail) => format!("cannot read file: {detail}"),
    })?;
    Ok(to_data_url(meta.format, &bytes))
}

/// Merge image data URLs into the final user turn as content parts,/// preserving the assembled text ahead of them. Pure — unit-tested.
/// Returns false (leaving `messages` untouched) when there is nothing to
/// attach to: empty message list or a non-user tail.
fn attach_images_to_last_message(messages: &mut [ChatMessage], urls: Vec<String>) -> bool {
    if urls.is_empty() {
        return false;
    }
    let Some(last) = messages.last_mut() else {
        return false;
    };
    if last.role != "user" {
        return false;
    }
    let text = last.text_content();
    let mut parts = Vec::with_capacity(urls.len() + 1);
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    parts.extend(urls.into_iter().map(|url| ContentPart::ImageUrl {
        image_url: ImageUrlPart { url, detail: None },
    }));
    last.content = MessageContent::Parts(parts);
    true
}

/// Maximum finding lines per `/advise` report before an overflow note.
pub const ADVISE_LINE_CAP: usize = 50;

/// Render the `/advise` report: a summary head line plus one line per
/// finding, optionally narrowed to files containing `filter`. Pure —
/// unit-tested.
pub fn format_advise_report(
    all: &[xencode_context_rs::Advice],
    filter: Option<&str>,
) -> Vec<String> {
    if all.is_empty() {
        return vec![
            "🔍 No findings — no cycles, hubs, orphans, or broken imports in the snapshot."
                .to_string(),
        ];
    }
    let shown: Vec<&xencode_context_rs::Advice> = all
        .iter()
        .filter(|a| filter.is_none_or(|f| a.file.contains(f)))
        .collect();
    if shown.is_empty() {
        return vec![format!(
            "🔍 No findings matching `{}` ({} total — drop the filter to see all).",
            filter.unwrap_or(""),
            all.len()
        )];
    }
    let mut broken = 0usize;
    let mut cycles = 0usize;
    let mut hubs = 0usize;
    let mut orphans = 0usize;
    let mut affected = 0usize;
    for a in &shown {
        match a.kind {
            xencode_context_rs::AdviceKind::BrokenImport => broken += 1,
            xencode_context_rs::AdviceKind::Cycle => cycles += 1,
            xencode_context_rs::AdviceKind::AffectedDependent => affected += 1,
            xencode_context_rs::AdviceKind::Hub => hubs += 1,
            xencode_context_rs::AdviceKind::Orphan => orphans += 1,
        }
    }
    let pl = |n: usize| if n == 1 { "" } else { "s" };
    let scope = if shown.len() == all.len() {
        String::new()
    } else {
        format!(" ({} total — drop the filter to see all)", all.len())
    };
    let mut out = vec![format!(
        "🔍 {} finding{} — {} broken import{}, {} cycle{}, {} hub{}, {} orphan{}, {} affected dependent{}{}:",
        shown.len(),
        pl(shown.len()),
        broken,
        pl(broken),
        cycles,
        pl(cycles),
        hubs,
        pl(hubs),
        orphans,
        pl(orphans),
        affected,
        pl(affected),
        scope,
    )];
    for a in shown.iter().take(ADVISE_LINE_CAP) {
        out.push(a.message.clone());
    }
    if shown.len() > ADVISE_LINE_CAP {
        out.push(format!(
            "… +{} more (narrow with /advise <path>).",
            shown.len() - ADVISE_LINE_CAP
        ));
    }
    out
}

/// Walk `root` and stream what the scanner actually found: one
/// `[SECURITY]finding:` per hit, `[SECURITY]progress:` per scanned file,
/// `[SECURITY]note:` for anything the panel should say, and a final
/// `[SECURITY]done:` carrying the true totals. A walker or read failure is
/// reported as `[SECURITY]failed:` in the scanner's own words.
async fn run_security_scan(root: std::path::PathBuf, tx: mpsc::UnboundedSender<String>) {
    let walked_root = root.clone();
    let walked = tokio::task::spawn_blocking(move || {
        let options = xencode_context_rs::ScanOptions::default();
        xencode_context_rs::scanner::scan_tree(&walked_root, &options)
    })
    .await;
    let outcome = match walked {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => {
            let _ = tx.send(format!("[SECURITY]failed:{}", e));
            return;
        }
        Err(e) => {
            let _ = tx.send(format!("[SECURITY]failed:scan thread died: {}", e));
            return;
        }
    };
    let total = outcome.files.len();
    let _ = tx.send(format!(
        "[SECURITY]note:{} files · {} listed as secret · {} skipped",
        outcome.files.len(),
        outcome.secret_files.len(),
        outcome.skipped
    ));

    // The walker counts secret-named files in `files` too and never reads them,
    // so they are reported as a class instead of being opened line by line.
    let mut reported = 0usize;
    let mut shown = 0usize;
    let mut by_severity = [0u32; 4]; // critical, high, medium, low
    let mut bump = |severity: &str| match severity {
        "Critical" => by_severity[0] += 1,
        "High" => by_severity[1] += 1,
        "Medium" => by_severity[2] += 1,
        _ => by_severity[3] += 1,
    };
    for path in &outcome.secret_files {
        reported += 1;
        bump("Medium");
        if shown < FINDINGS_CAP {
            shown += 1;
            let _ = tx.send(format!(
                "[SECURITY]finding:Medium|secret-file|{}|listed by the walker, never read — check whether it belongs in the tree",
                path
            ));
        }
    }

    let mut scanned = 0usize;
    let mut unreadable = 0usize;
    for entry in &outcome.files {
        if entry.is_binary || entry.is_secret {
            continue;
        }
        scanned += 1;
        let path = root.join(&entry.path);
        let findings = match xencode_analysis_rs::VulnerabilityScanner::scan_file(&path) {
            Ok(findings) => findings,
            Err(_) => {
                unreadable += 1;
                continue;
            }
        };
        for finding in findings {
            let severity = format!("{:?}", finding.severity);
            reported += 1;
            bump(&severity);
            if shown >= FINDINGS_CAP {
                continue;
            }
            shown += 1;
            let cwe = finding
                .cwe_id
                .as_ref()
                .map(|c| format!(" ({})", c))
                .unwrap_or_default();
            let _ = tx.send(format!(
                "[SECURITY]finding:{}|{}|{}:{}|{}{} — {}",
                severity,
                finding.finding_type,
                finding.file_path,
                finding.line_number,
                finding.message,
                cwe,
                finding.recommendation
            ));
        }
        let _ = tx.send(format!(
            "[SECURITY]progress:{:.2}",
            scanned as f64 / total.max(1) as f64
        ));
    }

    if reported > shown {
        let _ = tx.send(format!(
            "[SECURITY]note:list capped at {} findings — the severity totals are the full scan",
            FINDINGS_CAP
        ));
    }
    if unreadable > 0 {
        let _ = tx.send(format!(
            "[SECURITY]note:{} files could not be read and were skipped",
            unreadable
        ));
    }
    let _ = tx.send(format!(
        "[SECURITY]done:{} findings across {} files|{},{},{},{}",
        reported, scanned, by_severity[0], by_severity[1], by_severity[2], by_severity[3]
    ));
}

fn format_uptime(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    match (s / 3600, (s % 3600) / 60) {
        (h, m) if h > 0 => format!("{}h {}m", h, m),
        (0, m) if m > 0 => format!("{}m {}s", m, s % 60),
        _ => format!("{}s", s),
    }
}

/// Jiffies in `/proc/self/stat` are clock ticks; 100/s is the kernel's fixed
/// USER_HZ for that file, so it is a divisor, not a guess about this machine.
const CLOCK_TICKS_PER_SEC: f64 = 100.0;

/// Read `utime + stime` (field 14 + 15) for this process, in seconds. The
/// command name (field 2) can contain spaces and parentheses, so the fields
/// are taken after the closing paren rather than by naive splitting.
fn process_cpu_secs() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/self/stat").ok()?;
    let tail = text.rsplit_once(')')?.1;
    let mut fields = tail.split_whitespace();
    // tail starts at field 3 (state); utime is field 14 → 12th token, stime 13th.
    let utime: f64 = fields.nth(11)?.parse().ok()?;
    let stime: f64 = fields.next()?.parse().ok()?;
    Some((utime + stime) / CLOCK_TICKS_PER_SEC)
}

/// Resident set of this process in MB (`/proc/self/statm` field 2, in 4 KiB
/// pages) and total system memory in MB from `/proc/meminfo`.
fn process_memory_mb() -> (Option<f64>, Option<f64>) {
    let rss = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|text| {
            text.split_whitespace()
                .nth(1)?
                .parse::<f64>()
                .ok()
                .map(|pages| pages * 4096.0 / 1048576.0)
        });
    let total = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.starts_with("MemTotal:"))?
                .split_whitespace()
                .nth(1)?
                .parse::<f64>()
                .ok()
                .map(|kib| kib / 1024.0)
        });
    (rss, total)
}

/// Sample this process and the persisted per-request metrics, streaming the
/// same `[PROFILER]` protocol the panel already speaks. CPU is a rate, so it
/// needs two reads of `/proc` with a real interval between them.
async fn run_profiler(xencode: std::path::PathBuf, tx: mpsc::UnboundedSender<String>) {
    let sampled = tokio::task::spawn_blocking(move || {
        let before = process_cpu_secs();
        std::thread::sleep(std::time::Duration::from_millis(PROFILER_SAMPLE_MS));
        let after = process_cpu_secs();
        let cpu = match (before, after) {
            (Some(b), Some(a)) if b <= a => {
                Some((a - b) / (PROFILER_SAMPLE_MS as f64 / 1000.0) * 100.0)
            }
            _ => None,
        };
        let (rss, total) = process_memory_mb();
        let rows = xencode_context_rs::read_metrics(&xencode);
        (cpu, rss, total, rows)
    })
    .await;

    let (cpu, rss, total, rows) = match sampled {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(format!("[PROFILER]failed:sample thread died: {}", e));
            return;
        }
    };

    match cpu {
        Some(v) => {
            let _ = tx.send(format!("[PROFILER]gauge:cpu|{:.1}", v));
            let _ = tx.send(format!(
                "[PROFILER]row:process|cpu over {} ms|{:.1}%",
                PROFILER_SAMPLE_MS, v
            ));
        }
        None => {
            let _ = tx.send(
                "[PROFILER]note:cpu unavailable — /proc/self/stat could not be read".to_string(),
            );
        }
    }
    match rss {
        Some(v) => {
            let _ = tx.send(format!("[PROFILER]gauge:mem|{:.1}", v));
            if let Some(t) = total {
                let _ = tx.send(format!("[PROFILER]gauge:memtotal|{:.0}", t));
            }
            let _ = tx.send(format!(
                "[PROFILER]row:process|resident set|{:.1} MB{}",
                v,
                match total {
                    Some(t) => format!(" of {:.0} MB", t),
                    None => String::new(),
                }
            ));
        }
        None => {
            let _ = tx.send(
                "[PROFILER]note:memory unavailable — /proc/self/statm could not be read"
                    .to_string(),
            );
        }
    }

    if rows.is_empty() {
        let _ = tx
            .send("[PROFILER]note:no metrics.jsonl yet — a llama.cpp turn records one".to_string());
    } else {
        let _ = tx.send(format!(
            "[PROFILER]row:metrics|recorded turns|{}",
            rows.len()
        ));
        for r in rows.iter().rev().take(PROFILER_METRIC_ROWS) {
            let _ = tx.send(format!(
                "[PROFILER]row:{}|turn {}|{}% kv · {} prompt · {:.0} tok/s · {} files",
                r.profile,
                format_row_time(r.ts_unix_ms),
                (r.kv_reuse_ratio() * 100.0) as u64,
                r.prompt_tokens,
                r.generation_tok_s,
                r.retrieved_files
            ));
        }
    }
    let _ = tx.send("[PROFILER]done".to_string());
}

/// Walk the workspace and send a row per language the walker actually saw:
/// files, lines, and the share of the project's lines that language holds.
/// Nothing is estimated here — a file the walk refused to read counts as a file
/// and contributes no lines, and that is said out loud in a note.
async fn run_language_scan(root: std::path::PathBuf, tx: mpsc::UnboundedSender<String>) {
    let walked_root = root.clone();
    let walked = tokio::task::spawn_blocking(move || {
        let options = xencode_context_rs::ScanOptions::default();
        xencode_context_rs::scanner::scan_tree(&walked_root, &options)
    })
    .await;
    let outcome = match walked {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => {
            let _ = tx.send(format!("[LANG]failed:{e}"));
            return;
        }
        Err(e) => {
            let _ = tx.send(format!("[LANG]failed:scan thread died: {e}"));
            return;
        }
    };

    let mut per_lang: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
    for entry in &outcome.files {
        let slot = per_lang
            .entry(entry.language.as_str().to_string())
            .or_insert((0, 0));
        slot.0 += 1;
        slot.1 += entry.loc;
    }
    let total_lines: u64 = per_lang.values().map(|(_, lines)| *lines).sum();
    let mut rows: Vec<(String, u64, u64)> = per_lang
        .into_iter()
        .map(|(language, (files, lines))| (language, files, lines))
        .collect();
    // Biggest first; ties by name, so the order is the same every run.
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    for (language, files, lines) in rows {
        let share = if total_lines == 0 {
            0.0
        } else {
            lines as f64 * 100.0 / total_lines as f64
        };
        let row = serde_json::json!({
            "language": language, "files": files, "lines": lines,
            "share": (share * 10.0).round() / 10.0,
        });
        let _ = tx.send(format!("[LANG]row:{row}"));
    }

    if outcome.files.is_empty() {
        let _ = tx.send(format!(
            "[LANG]note:the walk found no files under {}",
            root.display()
        ));
    } else {
        let _ = tx.send(format!(
            "[LANG]note:{} files · {} lines · {} skipped by ignore rules",
            outcome.files.len(),
            total_lines,
            outcome.skipped
        ));
    }
    if !outcome.secret_files.is_empty() || !outcome.binary_files.is_empty() {
        let _ = tx.send(format!(
            "[LANG]note:{} listed as secret, {} binary — counted as files, never read, so they add no lines",
            outcome.secret_files.len(),
            outcome.binary_files.len()
        ));
    }
    let unread = outcome
        .files
        .iter()
        .filter(|e| !e.is_binary && !e.is_secret && e.loc == 0)
        .count();
    if unread > 0 {
        let _ = tx.send(format!(
            "[LANG]note:{unread} file(s) counted but unreadable — 0 lines"
        ));
    }
    let _ = tx.send("[LANG]done".to_string());
}

/// One capture session, start to finish, on a blocking thread: read the
/// recorder, report a level per chunk, save the WAV, then transcribe it only if
/// an engine is installed. Every token it sends describes something that
/// happened; the failure paths are the interesting ones.
fn run_voice_capture(
    recorder: &std::path::Path,
    args: &[std::ffi::OsString],
    clip_dir: &std::path::Path,
    stop: &std::sync::atomic::AtomicBool,
    muted: &std::sync::atomic::AtomicBool,
    tx: &mpsc::UnboundedSender<String>,
) -> std::io::Result<()> {
    let cap = crate::voice::capture(recorder, args, stop, muted, |level, bytes| {
        let _ = tx.send(format!("[VOICE]level:{level:.4}|{bytes}"));
    });

    if let Some(err) = &cap.error {
        let _ = tx.send(format!("[VOICE]err:{err}"));
        return Ok(());
    }
    if cap.pcm.is_empty() {
        let _ = tx.send(
            "[VOICE]note:The recorder sent no audio. Check the input device, or unmute with m."
                .to_string(),
        );
        return Ok(());
    }

    let ms = cap.ms();
    let name = format!(
        "clip-{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let clip = clip_dir.join(&name);
    std::fs::write(
        &clip,
        crate::voice::wav_bytes(&cap.pcm, crate::voice::SAMPLE_RATE),
    )?;
    let _ = tx.send(format!(
        "[VOICE]peak:{:.4}",
        cap.levels.iter().cloned().fold(0.0f64, f64::max)
    ));
    let _ = tx.send(format!("[VOICE]clip:{}|{ms}", clip.display()));

    match crate::voice::find_transcriber() {
        Some(bin) => {
            let _ = tx.send("[VOICE]status:processing".to_string());
            match crate::voice::transcribe(&bin, &clip) {
                Ok(text) => {
                    let _ = tx.send(format!(
                        "[VOICE]transcript:{}",
                        crate::agent_tools::truncate_one_line(&text, 500)
                    ));
                }
                Err(e) => {
                    let _ = tx.send(format!("[VOICE]err:{e}"));
                }
            }
        }
        None => {
            let _ = tx.send(format!(
                "[VOICE]note:{}",
                crate::voice::missing_transcriber_note(&clip)
            ));
        }
    }
    Ok(())
}

/// `level:<rms>|<bytes>` from the capture thread. `None` on anything else, so a
/// malformed reading cannot zero the meter.
fn parse_voice_level(body: &str) -> Option<(f64, usize)> {
    let (level, bytes) = body.split_once('|')?;
    Some((level.parse().ok()?, bytes.parse().ok()?))
}

/// The token budgets `←`/`→` step a custom-model profile through. A ladder
/// rather than free-form entry because these are the numbers worth sending.
const MODEL_TOKEN_STEPS: [u32; 8] = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
/// Lessons the Learning panel queues from the project index, and how much of a
/// file it puts on screen and into the prompt. Both caps exist because the
/// panel is a popup, not a pager.
const LEARN_LESSON_CAP: usize = 5;
const LEARN_SOURCE_CAP: usize = 4_000;

/// What the model sent back for one lesson. `answer` is the model's own key,
/// which is the only reason the panel can call anything correct.
struct LearnedLesson {
    explain: Vec<String>,
    question: String,
    options: Vec<String>,
    answer: usize,
    why: String,
}

/// The lesson queue: files `.xencode/index/symbols.json` says declare
/// something, most declarations first and ties by path so the order is the same
/// every run. `Err` carries the reason there is nothing to teach from, which is
/// what the panel shows instead of a lesson.
fn learning_lessons(root: &std::path::Path) -> Result<Vec<(String, Vec<String>)>, String> {
    let xencode = root.join(".xencode");
    if !xencode_context_rs::file_index_path(&xencode).is_file() {
        return Err("No project index — run /init first, then Enter again.".to_string());
    }
    let symbols: std::collections::BTreeMap<String, xencode_context_rs::PerFileSymbols> =
        xencode_context_rs::read_json(&xencode_context_rs::symbols_json_path(&xencode))
            .unwrap_or_default();
    let mut lessons: Vec<(String, Vec<String>)> = symbols
        .iter()
        .filter_map(|(path, syms)| {
            let mut declared: Vec<String> = syms
                .structs
                .iter()
                .map(|name| format!("struct {name}"))
                .collect();
            declared.extend(syms.functions.iter().map(|name| format!("fn {name}")));
            (!declared.is_empty()).then(|| (path.clone(), declared))
        })
        .collect();
    lessons.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    lessons.truncate(LEARN_LESSON_CAP);
    if lessons.is_empty() {
        return Err(
            "The index lists no file that declares a struct or function — nothing real to teach."
                .to_string(),
        );
    }
    Ok(lessons)
}

/// Cut `text` to at most `max` bytes, at a line boundary.
fn cap_at_line(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let window = &text[..max];
    match window.rfind('\n') {
        Some(keep) => text[..keep].to_string(),
        None => window.to_string(),
    }
}

/// Read the model's lesson out of its reply. Fences and surrounding prose are
/// tolerated; a reply that carries no complete quiz — fewer than two options,
/// an answer index past the end, no explanation — is `None`, so the panel can
/// report it instead of guessing.
fn parse_lesson_quiz(text: &str) -> Option<LearnedLesson> {
    let cleaned = text.replace("```json", "").replace("```", "");
    let start = cleaned.find('{')?;
    let end = cleaned.rfind('}')?;
    if end <= start {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(&cleaned[start..=end]).ok()?;
    let strings = |key: &str| -> Vec<String> {
        let Some(items) = value[key].as_array() else {
            return Vec::new();
        };
        items
            .iter()
            .map(|item| match item {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let explain = strings("explain");
    let options = strings("options");
    let question = value["question"].as_str().unwrap_or_default().trim();
    // Weak models quote the index; accept "1" as readily as 1.
    let answer = value["answer"].as_u64().or_else(|| {
        value["answer"]
            .as_str()
            .and_then(|s| s.trim().parse::<u64>().ok())
    })? as usize;
    if question.is_empty()
        || explain.is_empty()
        || options.len() < 2
        || options.len() > 4
        || answer >= options.len()
    {
        return None;
    }
    Some(LearnedLesson {
        explain,
        question: question.to_string(),
        options,
        answer,
        why: value["why"].as_str().unwrap_or_default().trim().to_string(),
    })
}

/// Commands the terminal panel will offer at once. More than a screenful is
/// noise, so this is both what the model is asked for and what it gets.
const TERM_SUGGESTION_CAP: usize = 8;

/// Commands the panel will never label safe, whatever the model claimed. The
/// model's risk label is a hint; this list can only raise the warning, never
/// lower one.
const DESTRUCTIVE_PATTERNS: &[&str] = &[
    "rm -rf",
    "rm -fr",
    "sudo ",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "chmod -R",
    "chown -R",
    "--force",
    "docker system prune",
    "DROP TABLE",
    "shutdown",
    "reboot",
    "killall",
];

/// Parse the model's reply into commands. Fences, prose and a bare object
/// (instead of an array) are all tolerated; a reply with no commands in it
/// yields an empty list, which the panel reports rather than papers over.
fn parse_term_suggestions(text: &str) -> Vec<(String, String, String)> {
    let cleaned = text.replace("```json", "").replace("```", "");
    let parsed: Option<serde_json::Value> = match (cleaned.find('['), cleaned.rfind(']')) {
        (Some(start), Some(end)) if end > start => serde_json::from_str(&cleaned[start..=end]).ok(),
        _ => None,
    };
    let values = match parsed {
        Some(serde_json::Value::Array(values)) => values,
        Some(object @ serde_json::Value::Object(_)) => vec![object],
        _ => return Vec::new(),
    };
    let mut out = Vec::new();
    for value in values {
        let command = value["command"].as_str().unwrap_or_default().trim();
        if command.is_empty() {
            continue;
        }
        let claimed = value["risk"]
            .as_str()
            .unwrap_or("safe")
            .to_ascii_lowercase();
        let destructive = claimed.contains("destruct")
            || claimed.contains("danger")
            || DESTRUCTIVE_PATTERNS.iter().any(|p| command.contains(p));
        let why = value["why"]
            .as_str()
            .or_else(|| value["explanation"].as_str())
            .unwrap_or_default()
            .trim()
            .to_string();
        out.push((
            command.to_string(),
            if destructive { "destructive" } else { "safe" }.to_string(),
            why,
        ));
        if out.len() == TERM_SUGGESTION_CAP {
            break;
        }
    }
    out
}

/// Metrics rows are stamped in epoch millis; rendered as UTC wall time without
/// pulling in a date library. A row that never got a timestamp says so rather
/// than showing a fake clock.
fn format_row_time(ts_unix_ms: u64) -> String {
    if ts_unix_ms == 0 {
        return "untimed".to_string();
    }
    let secs = ts_unix_ms / 1000;
    format!(
        "{:02}:{:02}:{:02}",
        (secs / 3600) % 24,
        (secs / 60) % 60,
        secs % 60
    )
}

/// One non-streaming provider request, built from the session config: the same
/// clients, keys and llama.cpp options a chat turn uses. A panel that needs a
/// single answer (terminal assistant, translation) asks through this instead of
/// carrying its own copy of the plumbing.
#[derive(Clone)]
struct SingleShot {
    model: String,
    ollama_url: String,
    llama_cpp_url: String,
    timeout: u64,
    openrouter_key: Option<String>,
    qwen_key: Option<String>,
    gemini_key: Option<String>,
    llama_opts: LlamaCppOptions,
}

impl SingleShot {
    fn from_config(config: &XencodeConfig) -> Self {
        Self {
            model: config.default_model.clone(),
            ollama_url: config.ollama_url.clone(),
            llama_cpp_url: config.llama_cpp_url.clone(),
            timeout: config.response_timeout,
            openrouter_key: config.api_keys.openrouter_api_key.clone(),
            qwen_key: config.api_keys.qwen_api_key.clone(),
            gemini_key: config.api_keys.google_gemini_api_key.clone(),
            llama_opts: LlamaCppOptions {
                temperature: config.llama_cpp_temperature,
                top_k: config.llama_cpp_top_k,
                min_p: config.llama_cpp_min_p,
                max_tokens: config.llama_cpp_max_tokens,
                grammar: None,
                json_schema: None,
                mirostat: None,
            },
        }
    }

    /// `Err` carries the provider's own words, because that is what the panel
    /// shows — a generic "translation failed" would hide why.
    async fn ask(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let client = OllamaClient::new(&self.ollama_url, self.timeout);
        let llama_client = LlamaCppClient::new(&self.llama_cpp_url, self.timeout);
        let manager = ProviderManager::new(
            client,
            self.openrouter_key.clone(),
            self.qwen_key.clone(),
            self.gemini_key.clone(),
            None,
        )
        .with_llama_cpp(llama_client);
        manager
            .generate_with_options(&self.model, messages, Some(&self.llama_opts))
            .await
            .map_err(|e| format!("{} said: {}", self.model, e))
    }
}

/// The frozen system head a chat turn sends (workspace instructions + anchor),
/// so a panel's one-shot request starts from the same prefix and stays cheap.
fn one_shot_messages(root: &std::path::Path, prompt: String) -> Vec<ChatMessage> {
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
    let anchor =
        std::fs::read_to_string(root.join(xencode_context_rs::XENCODE_DIR).join("anchor.md")).ok();
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: xencode_context_rs::stable_system_text(
                CTX_SYSTEM,
                agents.as_deref(),
                anchor.as_deref(),
            )
            .into(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: prompt.into(),
        },
    ]
}

impl<'a> App<'a> {
    pub fn new() -> Self {
        let config = XencodeConfig::load().unwrap_or_default();
        let mut memory = ConversationMemory::with_persistence(config.max_memory_items)
            .unwrap_or_else(|_| ConversationMemory::new(50));
        memory.start_session(None);
        Self::with_config_and_memory(config, memory)
    }

    /// Isolated app for tests: default config, non-persistent conversation
    /// memory, and config writes disabled. `App::new()` reads *and writes* the
    /// user's real `<config dir>/conversation_memory.json` — the restored
    /// history makes transcript assertions non-deterministic and the writes
    /// pollute the user's home — so no test may call it.
    pub fn for_tests() -> Self {
        let mut app =
            Self::with_config_and_memory(XencodeConfig::default(), ConversationMemory::new(50));
        app.persist_config = false;
        app
    }

    fn with_config_and_memory(config: XencodeConfig, memory: ConversationMemory) -> Self {
        let _client = OllamaClient::new(&config.ollama_url, config.response_timeout);
        // `config` moves into the struct below; the panel needs its own copy.
        let model_profiles = config.model_profiles.clone();

        let scan_opts = ScanOptions {
            max_depth: Some(5),
            include_hidden: false,
            excluded_dirs: vec![
                ".git".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "__pycache__".to_string(),
                ".pytest_cache".to_string(),
                ".venv".to_string(),
            ],
        };
        let tree = scan_workspace(".", &scan_opts).unwrap_or_default();
        let file_tree: Vec<String> = tree
            .into_iter()
            .map(|f| f.path.display().to_string())
            .collect();

        let available_models = if !config.default_model.is_empty() {
            vec![config.default_model.clone()]
        } else {
            vec!["qwen2.5:7b".to_string()]
        };
        let selected_model = 0;
        let theme = ThemeColors::get(&config.active_theme);

        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let git_status = git_status_map();
        let git_branch = Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "main".to_string());

        let mut editor = TextArea::default();
        editor.set_line_number_style(
            ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
        );

        let now = current_timestamp();

        let mut app = Self {
            focus: FocusArea::ChatInput,
            chat_input: TextArea::default(),
            input_mode: InputMode::Normal,
            messages: Vec::new(),
            chat_scroll: 0,
            input_history: Vec::new(),
            history_index: None,
            history_draft: String::new(),
            file_tree,
            selected_file: 0,
            attached_files: HashSet::new(),
            opened_file: None,
            editor,
            editor_dirty: false,
            git_status,
            git_branch,
            available_models,
            selected_model,
            is_generating: false,
            is_reviewing: false,
            code_review_output: String::new(),
            review_dash: crate::review::ReviewDashboard::new(),
            tasks_selected: 0,
            tasks_detail: false,
            tasks_scroll: 0,
            worktrees: Vec::new(),
            worktree_dirty: Vec::new(),
            worktree_selected: 0,
            worktree_prompt: crate::focus::WorktreePrompt::None,
            worktree_path_buf: String::new(),
            worktree_branch_buf: String::new(),
            worktree_status: String::new(),
            advise_items: Vec::new(),
            advise_selected: 0,
            advise_detail: false,
            advise_scroll: 0,
            advise_status: String::new(),
            commit_message: String::new(),
            commit_cursor: 0,
            spinner_tick: 0,
            theme,
            persist_config: true,
            config,
            show_terminal: false,
            last_body_focus: FocusArea::ChatInput,
            last_layout: crate::layout::BodyLayout::default(),
            agent_grants: Arc::new(std::sync::Mutex::new(Vec::new())),
            checkpoints: Arc::new(crate::agent_tools::CheckpointStore::new()),
            agent_plan: crate::agent_tools::new_plan_handle(),
            mcp: Arc::new(crate::mcp::McpHub::new()),
            plan_pinned: false,
            approval_queue: std::collections::VecDeque::new(),
            approval_scroll: 0,
            approval_tx,
            approval_rx: Some(approval_rx),
            memory,
            feature_nav_selected: 0,
            session_start_time: now,
            ollama_health_entries: HashMap::new(),
            last_health_check: 0.0,
            health_check_in_progress: false,
            total_llm_calls: 0,
            average_latency: 0.0,
            bytebot_command: String::new(),
            bytebot_cursor: 0,
            bytebot_steps: Vec::new(),
            bytebot_progress: 0.0,
            bytebot_running: false,
            bytebot_log: Vec::new(),
            bytebot_history: Vec::new(),
            spawns: Vec::new(),
            spawn_next_id: 1,
            context_hint_shown: false,
            init_running: false,
            init_progress: 0.0,
            init_steps: Vec::new(),
            init_log: Vec::new(),
            init_visible: false,
            help_visible: false,
            help_scroll: 0,
            toasts: Vec::new(),
            init_cancel: Arc::new(AtomicBool::new(false)),
            collab_session_active: false,
            collab_session_id: String::new(),
            collab_members: Vec::new(),
            collab_sync_status: "disconnected".to_string(),
            collab_last_sync: 0.0,
            collab_activity_log: Vec::new(),
            collab_server_url: "http://127.0.0.1:8765".to_string(),
            collab_username: std::env::var("USER").unwrap_or_else(|_| "you".to_string()),
            collab_worker: None,
            collab_error: String::new(),
            collab_editing: false,
            collab_field: crate::focus::CollabField::Server,

            voice_active: false,
            voice_status: "idle".to_string(),
            voice_level: 0.0,
            voice_peak: 0.0,
            voice_pcm_bytes: 0,
            voice_transcript: Vec::new(),
            voice_note: String::new(),
            voice_clip: None,
            voice_recorder: String::new(),
            voice_muted: false,
            voice_busy: false,
            voice_stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            voice_mute_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            voice_root: std::path::PathBuf::new(),

            term_asst_busy: false,
            term_asst_query: String::new(),
            // The panel opens ready to be typed into: it has nothing to show
            // until a query is asked.
            term_asst_typing: true,
            term_asst_suggestions: Vec::new(),
            term_asst_selected: 0,
            term_asst_output: String::new(),
            term_asst_history: Vec::new(),
            term_risk_filter: "All".to_string(),

            sec_scan_active: false,
            sec_scan_path: String::new(),
            sec_scan_results: Vec::new(),
            sec_scan_summary: (0, 0, 0, 0),
            sec_scan_progress: 0.0,
            sec_scan_log: Vec::new(),
            sec_filter_severity: "All".to_string(),
            sec_sort_mode: "severity".to_string(),

            profiler_active: false,
            profiler_running: false,
            profiler_rows: Vec::new(),
            profiler_notes: Vec::new(),
            profiler_gauge_cpu: None,
            profiler_gauge_mem: None,
            profiler_gauge_mem_total: None,
            profiler_gauge_latency: None,

            model_profiles,
            models_selected: 0,
            models_dirty: false,
            models_status: String::new(),
            models_busy: false,

            learn_active: false,
            learn_root: std::path::PathBuf::new(),
            learn_lessons: Vec::new(),
            learn_current_lesson: 0,
            learn_total_lessons: 0,
            learn_lesson_title: String::new(),
            learn_content: Vec::new(),
            learn_code_example: String::new(),
            learn_status: String::new(),
            learn_busy: false,
            learn_quiz_active: false,
            learn_quiz_question: String::new(),
            learn_quiz_options: Vec::new(),
            learn_quiz_selected: 0,
            learn_quiz_answered: false,
            learn_quiz_correct: false,
            learn_quiz_answer: None,
            learn_explain: Vec::new(),
            learn_quiz_why: String::new(),

            lang_busy: false,
            lang_scan_path: String::new(),
            lang_detection_results: Vec::new(),
            lang_notes: Vec::new(),
            lang_translate_input: String::new(),
            lang_translate_output: String::new(),
            lang_translate_source: "auto".to_string(),
            lang_translate_target: "English".to_string(),
            lang_translate_error: false,
            lang_editing: None,

            provider_health_scroll: 0,
            security_scroll: 0,
            review_scroll: 0,

            settings_cursor: 0,
            settings_reset_active: false,
            settings_url_editing: false,
            settings_url_buffer: String::new(),
            settings_url_cursor: 0,

            llamacpp_editing: false,
            llamacpp_path_buffer: String::new(),
            llamacpp_path_cursor: 0,
            llamacpp_action_msg: String::new(),
            sampling_temp_editing: false,
            sampling_temp_buffer: String::new(),
            sampling_int_editing: false,
            sampling_int_buffer: String::new(),

            last_llamacpp_timings: None,
            llama_process: None,
            llama_cancel: Arc::new(AtomicBool::new(false)),
            task_runtime: crate::agent_tools::new_task_runtime(),
            last_ctx_total_tokens: 0,
            last_ctx_retrieved_files: 0,
        };
        app.style_chat_input();

        // Seed initial health entries for configured providers
        app.ollama_health_entries.insert(
            "ollama".to_string(),
            (HealthStatus::Unknown.to_string(), 0.0, None),
        );
        app.ollama_health_entries.insert(
            "openrouter".to_string(),
            (
                if app.config.api_keys.openrouter_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.openrouter_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "qwen".to_string(),
            (
                if app.config.api_keys.qwen_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.qwen_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "gemini".to_string(),
            (
                if app.config.api_keys.google_gemini_api_key.is_some() {
                    HealthStatus::Unknown.to_string()
                } else {
                    HealthStatus::Error.to_string()
                },
                0.0,
                if app.config.api_keys.google_gemini_api_key.is_none() {
                    Some("API key not configured".to_string())
                } else {
                    None
                },
            ),
        );
        app.ollama_health_entries.insert(
            "llamacpp".to_string(),
            (HealthStatus::Unknown.to_string(), 0.0, None),
        );

        for msg in app.memory.get_context(10) {
            app.messages.push(UiMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }
        app
    }

    pub fn open_file_in_editor(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
                self.editor = TextArea::new(if lines.is_empty() {
                    vec![String::new()]
                } else {
                    lines
                });
                self.editor.set_line_number_style(
                    ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
                );
                self.opened_file = Some(path.to_string());
                self.editor_dirty = false;
            }
            Err(_) => {
                self.editor = TextArea::new(vec![format!("Unable to read file: {}", path)]);
                self.opened_file = None;
            }
        }
    }

    pub fn save_editor(&mut self) {
        if let Some(ref fp) = self.opened_file {
            let content: String = self.editor.lines().join("\n");
            match std::fs::write(fp, &content) {
                Ok(()) => self.editor_dirty = false,
                Err(e) => self.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!(
                        "⚠ Could not save {fp}: {e} — your edits are still in the editor."
                    ),
                }),
            }
        }
    }

    /// True when a Normal-mode text field owns the keyboard (GitCommit message,
    /// ByteBot command, settings URL buffer). Universal single-key shortcuts
    /// must not swallow the input itself (E2-06).
    pub fn text_entry_active(&self) -> bool {
        matches!(self.focus, FocusArea::GitCommit | FocusArea::ByteBotPanel)
            || (self.focus == FocusArea::Settings && self.settings_url_editing)
            || (self.focus == FocusArea::CollaborationHub && self.collab_editing)
    }

    pub fn refresh_git(&mut self) {
        self.git_status = git_status_map();
        if let Ok(output) = Command::new("git")
            .args(["branch", "--show-current"])
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                let branch = s.trim().to_string();
                if !branch.is_empty() {
                    self.git_branch = branch;
                }
            }
        }
    }

    /// The chat box is a plain textarea; only its colors follow the theme.
    /// Persist the working config to disk. One choke point for every
    /// settings write (H1-04).
    pub fn save_config(&mut self) {
        if self.persist_config {
            let _ = self.config.save();
        }
    }

    /// The agent tool-loop's approval mode, parsed from config with the
    /// strictest value as fallback (I1-01).
    pub fn agent_mode(&self) -> crate::agent_tools::ApprovalMode {
        crate::agent_tools::ApprovalMode::parse(&self.config.agent_approval)
    }

    /// Remember an "always allow for this session" approval answer. Never
    /// written to config — quitting revokes every grant.
    pub fn grant_tools_for_session(&mut self, class: crate::agent_tools::ToolClass) {
        if let Ok(mut grants) = self.agent_grants.lock() {
            if !grants.contains(&class) {
                grants.push(class);
            }
        }
    }

    /// The approval prompt the overlay shows right now, if any.
    pub fn pending_approval(&self) -> Option<&crate::agent_tools::ApprovalRequest> {
        self.approval_queue.front().map(|(request, _)| request)
    }

    /// Answer the frontmost prompt: wake the waiting tool task, apply the
    /// session grant for "always allow", and record the decision in the
    /// chat transcript using the same ⚙ grammar as the tool-loop lines.
    pub fn resolve_approval(&mut self, answer: crate::agent_tools::ApprovalAnswer) {
        let Some((request, responder)) = self.approval_queue.pop_front() else {
            return;
        };
        if answer == crate::agent_tools::ApprovalAnswer::ApprovedForSession {
            self.grant_tools_for_session(request.class);
        }
        let _ = responder.send(answer);
        self.approval_scroll = 0;
        self.messages.push(UiMessage {
            role: "system".to_string(),
            content: format!("⚙ {} · {}", request.summary, answer.tag()),
        });
    }

    pub(crate) fn style_chat_input(&mut self) {
        self.chat_input
            .set_style(ratatui::style::Style::default().fg(self.theme.fg));
        self.chat_input
            .set_cursor_line_style(ratatui::style::Style::default());
    }

    fn reset_chat_input(&mut self) {
        self.chat_input = TextArea::default();
        self.style_chat_input();
    }

    /// Replace the whole draft (history recall / slash completion).
    fn set_chat_text(&mut self, text: &str) {
        let lines: Vec<String> = if text.is_empty() {
            vec![String::new()]
        } else {
            text.lines().map(String::from).collect()
        };
        self.chat_input = TextArea::from(lines);
        self.style_chat_input();
        self.chat_input.move_cursor(CursorMove::Bottom);
        self.chat_input.move_cursor(CursorMove::End);
    }

    /// Alt+Up/Down: walk sent prompts; index past the ends restores the
    /// stashed draft. Plain Up/Down while editing stay textarea navigation.
    pub(crate) fn recall_history(&mut self, dir: i32) {
        if self.input_history.is_empty() {
            return;
        }
        let len = self.input_history.len() as i32;
        let current = self.history_index.map(|i| i as i32).unwrap_or(len);
        let next = (current + dir).clamp(0, len);
        if next == len {
            let draft = std::mem::take(&mut self.history_draft);
            self.history_index = None;
            self.set_chat_text(&draft);
        } else {
            if self.history_index.is_none() {
                self.history_draft = self.chat_input.lines().join("\n");
            }
            self.history_index = Some(next as usize);
            let text = self.input_history[next as usize].clone();
            self.set_chat_text(&text);
        }
    }

    pub(crate) fn push_toast(&mut self, kind: crate::toast::ToastKind, message: String) {
        crate::toast::push(&mut self.toasts, message, kind, current_timestamp());
    }

    /// Tab on a `/...` first line: complete the command token. Returns the
    /// completed token, or None when nothing can be completed (the caller
    /// then falls back to inserting spaces).
    pub(crate) fn complete_slash_draft(&mut self) -> bool {
        let draft = self.chat_input.lines().join("\n");
        if draft.lines().count() > 1 || !draft.starts_with('/') || draft.contains(' ') {
            return false;
        }
        if draft == "/" {
            self.push_toast(
                crate::toast::ToastKind::Info,
                "Commands: /init  /ctx  /advise  /bytebot  /spawn  /plan  /rewind  /mcp (Tab completes)"
                    .to_string(),
            );
            return true;
        }
        match complete_slash_token(&draft) {
            Some(done) => {
                self.set_chat_text(&format!("{done} "));
                true
            }
            None => false,
        }
    }

    pub fn submit_message(&mut self, tx: mpsc::UnboundedSender<String>) {
        let prompt = self.chat_input.lines().join("\n");
        if prompt.trim().is_empty() {
            return;
        }
        self.reset_chat_input();
        if self.input_history.last().is_none_or(|last| *last != prompt) {
            self.input_history.push(prompt.clone());
            if self.input_history.len() > INPUT_HISTORY_LIMIT {
                self.input_history.remove(0);
            }
        }
        self.history_index = None;
        self.history_draft.clear();

        self.messages.push(UiMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
        // Only real conversation goes to persistent memory: a slash command is
        // a local TUI verb, and persisting it replays it to the model (and into
        // restored transcripts) forever.
        if !prompt.starts_with('/') {
            self.memory.add_message("user", &prompt, None);
        }

        // Project context engine interception (/init, /init abort, /init status)
        if prompt.starts_with("/init") {
            self.handle_init_command(&prompt, tx);
            return;
        }

        // Context assembly interception (/ctx, /ctx status, /ctx track <path>)
        if prompt.starts_with("/ctx") {
            self.handle_ctx_command(&prompt, tx);
            return;
        }

        // Repository insights (/advise [filter])
        if prompt.starts_with("/advise") {
            self.handle_advise_command(&prompt, tx);
            return;
        }

        // Undo what the agent changed (/rewind [turns])
        if prompt == "/rewind" || prompt.starts_with("/rewind ") {
            self.handle_rewind_command(&prompt);
            return;
        }

        // The agent's todo list (/plan, /plan clear)
        if prompt == "/plan" || prompt.starts_with("/plan ") {
            self.handle_plan_command(&prompt);
            return;
        }

        // ByteBot: the same gated tool loop, reported to its own panel (I2-04)
        if prompt.starts_with("/bytebot") {
            let task = prompt
                .strip_prefix("/bytebot")
                .unwrap_or("")
                .trim()
                .to_string();
            if let Some(run) = self.arm_bytebot(&task) {
                tokio::spawn(agent_rounds(run, tx));
            }
            return;
        }

        // /spawn: a delegated subagent isolated in its own git worktree (I3-03)
        if prompt == "/spawn" || prompt.starts_with("/spawn ") {
            self.handle_spawn_command(&prompt, tx);
            return;
        }

        // MCP servers: connect the configured ones, or ask about/stop them
        // (I3-01). A declared server is only ever started here, on request.
        if prompt == "/mcp" || prompt.starts_with("/mcp ") {
            self.handle_mcp_command(&prompt, tx);
            return;
        }

        self.is_generating = true;

        // Normal LLM generation — project context is injected on every turn:
        // a byte-stable system head (KV-cacheable) + budgeted history turns +
        // retrieval/state/git riding in the final user turn (§10 tiers).
        //
        // The current prompt was just appended to memory above, so the tail
        // entry is popped back off: history holds prior turns only, and the
        // assembler places the prompt itself (unsqueezable) last.
        let mut history: Vec<(String, String)> = self
            .memory
            .get_context(26)
            .into_iter()
            .map(|m| (m.role, m.content))
            .collect();
        if history
            .last()
            .is_some_and(|(role, content)| role == "user" && content == &prompt)
        {
            history.pop();
        }
        let root = xencode_context_rs::default_root();
        let live = xencode_context_rs::collect_live_context(&root, &prompt, CTX_PROFILE);
        // Sorted for a deterministic prompt (and KV prefix) across turns.
        // Images ride as message parts, not inlined text: read_to_string
        // would silently drop them, and raw bytes would corrupt the prompt.
        let mut attached_paths: Vec<&String> = self.attached_files.iter().collect();
        attached_paths.sort();
        let mut attached_block = String::new();
        let mut attached_image_urls: Vec<String> = Vec::new();
        for path in attached_paths {
            if xencode_analysis_rs::is_image_path(std::path::Path::new(path)) {
                match encode_attached_image(path) {
                    Ok(url) => attached_image_urls.push(url),
                    Err(reason) => attached_block.push_str(&format!(
                        "<file path=\"{path}\">\n(image not sent: {reason})\n</file>\n\n"
                    )),
                }
            } else if xencode_context_rs::is_document_path(std::path::Path::new(path)) {
                attached_block.push_str(&doc_attach_block(path, &parse_attached_document(path)));
            } else {
                append_text_attachment(&mut attached_block, path);
            }
        }
        // The model's real window when known (Step 3 capabilities); unknown
        // routes defer to the profile default. The `/ctx` preview always
        // shows profile-default budgeting.
        let model = self.config.default_model.clone();
        let context_window = xencode_providers_rs::capabilities_for(&model).context_window;
        let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
            profile: CTX_PROFILE,
            context_window,
            system: CTX_SYSTEM,
            agents_md: live.agents_md.as_deref(),
            anchor_md: live.anchor_md.as_deref(),
            state_md: live.state_md.as_deref(),
            git_summary: &live.git_summary,
            retrieved: live.blocks,
            attached_block: &attached_block,
            history: &history,
            prompt: &prompt,
        });
        if !live.index_present && !self.context_hint_shown {
            self.context_hint_shown = true;
            self.messages.push(UiMessage {
                role: "system".to_string(),
                content: "Project index not found — run /init once for project-aware answers. Continuing with guidelines + history only.".to_string(),
            });
        }
        let mut context_messages: Vec<ChatMessage> = assembly
            .turns
            .into_iter()
            .map(|t| ChatMessage {
                role: t.role,
                content: t.content.into(),
            })
            .collect();
        // Teach the model the tool vocabulary (I1-04). Appended to the
        // assembled system turn so context assembly and its KV-cache
        // stability are untouched; only text-only system messages qualify.
        if let Some(system) = context_messages.first_mut().filter(|m| m.role == "system") {
            if let xencode_providers_rs::MessageContent::Text(text) = &mut system.content {
                text.push_str(crate::agent_tools::TOOL_HINT);
            }
        }
        // Attached images become content parts on the final user turn, in
        // sorted-path order (deterministic, KV-stable like the text block).
        // A `false` here means the turn was unusable — surface it in chat
        // rather than dropping the user's images silently.
        let images_pending = !attached_image_urls.is_empty();
        let images_attached =
            attach_images_to_last_message(&mut context_messages, attached_image_urls);
        if images_pending && !images_attached {
            self.messages.push(UiMessage {
                role: "system".to_string(),
                content: "⚠ Attached images could not be sent with this turn (no final user message) — they were dropped, not seen by the model.".to_string(),
            });
        }

        // Metrics for this real generation (reaches `/ctx kv` via [CTXSTATS]
        // in the drain loop, next to the llama.cpp [TIMINGS]).
        let xencode = root.join(xencode_context_rs::XENCODE_DIR);
        Self::record_ctx_metrics(
            &xencode,
            CTX_PROFILE,
            assembly.total_tokens,
            assembly.target_tokens,
            assembly.retrieved_included,
            assembly.soft_compaction_needed,
        );
        let _ = tx.send(format!(
            "[CTXSTATS]{}|{}",
            assembly.total_tokens.min(u32::MAX as u64),
            assembly.retrieved_included.min(u8::MAX as usize)
        ));

        let run = self.agent_run(LoopSink::Chat, context_messages);

        tokio::spawn(agent_rounds(run, tx));
    }

    /// This session's permission state: mode, shared grants, the overlay's
    /// channel, checkpoints and budgets. Every caller that runs a tool — the
    /// chat loop, ByteBot, a spawn, the terminal panel — goes through this, so
    /// there is exactly one policy in the app.
    fn approval_ctx(&self) -> crate::agent_tools::ApprovalCtx {
        crate::agent_tools::ApprovalCtx {
            mode: self.agent_mode(),
            grants: self.agent_grants.clone(),
            prompts: self.approval_tx.clone(),
            checkpoints: self.checkpoints.clone(),
            // One checkpoint group per user turn (I2-01): `/rewind` steps
            // back whole turns, not individual tool calls.
            turn: self.checkpoints.begin_turn(),
            // A 0-second budget would kill every command before it
            // produced output, so the floor is one second.
            command_timeout: self.config.agent_command_timeout.max(1),
            plan: self.agent_plan.clone(),
            mcp: self.mcp.clone(),
            hooks: self.config.agent_hooks.clone(),
        }
    }

    /// A tool-loop run carrying this session's providers, permission state,
    /// checkpoint group and budgets. Read at the moment a turn starts, so a
    /// settings change lands on the next turn; chat and ByteBot build the same
    /// run and differ only in `sink` (I2-04).
    fn agent_run(&self, sink: LoopSink, context_messages: Vec<ChatMessage>) -> AgentRun {
        AgentRun {
            sink,
            model: self.config.default_model.clone(),
            context_messages,
            approval: self.approval_ctx(),
            task_runtime: self.task_runtime.clone(),
            tool_root: xencode_context_rs::default_root(),
            // Keep at least one tool round; 0 would offer tools on no turn.
            max_rounds: self.config.agent_max_rounds.clamp(1, 64),
            fallback_models: self.config.agent_fallback_models.clone(),
            ollama_url: self.config.ollama_url.clone(),
            llama_cpp_url: self.config.llama_cpp_url.clone(),
            timeout: self.config.response_timeout,
            openrouter_key: self.config.api_keys.openrouter_api_key.clone(),
            qwen_key: self.config.api_keys.qwen_api_key.clone(),
            gemini_key: self.config.api_keys.google_gemini_api_key.clone(),
            llama_opts: LlamaCppOptions {
                temperature: self.config.llama_cpp_temperature,
                top_k: self.config.llama_cpp_top_k,
                min_p: self.config.llama_cpp_min_p,
                max_tokens: self.config.llama_cpp_max_tokens,
                grammar: None,
                json_schema: None,
                mirostat: None,
            },
        }
    }

    /// Send a load/unload/switch command to the llama.cpp server and report the
    /// result back through the channel.
    ///
    /// Commands:
    /// - "load"        : load `target` (a model id) or the configured GGUF path.
    /// - "switch"      : swap to `target` (a model id) — unloads first if we can.
    /// - "unload"      : unload whatever is loaded.
    pub fn llamacpp_control(
        &mut self,
        command: &str,
        target: Option<String>,
        tx: mpsc::UnboundedSender<String>,
    ) {
        let llamacpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let load_target = if let Some(t) = target {
            t
        } else {
            self.config.llama_cpp_model_path.clone()
        };

        self.llamacpp_action_msg = match command {
            "load" | "switch" => {
                if load_target.is_empty() {
                    "Set a GGUF model path first (Settings → Llama.cpp Model Path), or pick a llama.cpp model".to_string()
                } else {
                    "Requesting model switch...".to_string()
                }
            }
            "unload" => "Requesting model unload...".to_string(),
            _ => return,
        };

        let (url, path) = (llamacpp_url, load_target.clone());
        let is_unload = command == "unload";
        tokio::spawn(async move {
            let client = LlamaCppClient::new(&url, timeout);
            let result = if is_unload {
                client.unload_models().await
            } else {
                client.load_model(&path).await
            };
            let msg = match result {
                Ok(()) => {
                    let label = if is_unload {
                        "model unloaded".to_string()
                    } else {
                        format!("model '{path}' loaded")
                    };
                    format!("✅ llama.cpp: {label}")
                }
                Err(e) => format!("❌ llama.cpp: {e}"),
            };
            let _ = tx.send(format!("[LLAMACPP]{}", msg));
        });
    }

    pub fn append_generation(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_generating = false;
            self.total_llm_calls += 1;
            if let Some(last) = self.messages.last() {
                if last.role == "assistant" {
                    self.memory.add_message(
                        "assistant",
                        &last.content,
                        Some(self.config.default_model.clone()),
                    );
                }
            }
            return;
        }
        if let Some(last) = self.messages.last_mut() {
            if last.role == "assistant" && self.is_generating {
                last.content.push_str(text);
                return;
            }
        }
        self.messages.push(UiMessage {
            role: "assistant".to_string(),
            content: text.to_string(),
        });
    }

    pub fn append_review(&mut self, text: &str) {
        if text == "[DONE]" {
            self.is_reviewing = false;
        } else {
            self.code_review_output.push_str(text);
        }
    }

    /// Registry snapshot for the task panel's draw/keys. `None` while the
    /// chat tool loop holds the lock — registry ops never await mid-lock,
    /// so this window is effectively between keystrokes.
    pub fn tasks_snapshot(&self) -> Option<Vec<xencode_core_rs::TaskRecord>> {
        self.task_runtime.try_lock().ok().map(|m| m.list().to_vec())
    }

    /// `[TASKS]<verb>[|id]>` from the panel keys (D2-01): mutate the shared
    /// registry off the UI thread. No feedback line — the panel renders the
    /// registry live, so the new status is the feedback.
    pub fn handle_tasks_command(&mut self, body: &str) {
        let mut parts = body.split('|');
        let Some(id) = parts
            .next()
            .filter(|v| matches!(*v, "stop" | "rm"))
            .and_then(|_| parts.next())
            .and_then(|s| s.parse::<u64>().ok())
        else {
            return;
        };
        let is_stop = body.starts_with("stop");
        let rt = self.task_runtime.clone();
        tokio::spawn(async move {
            let mut m = rt.lock().await;
            if is_stop {
                let _ = m.stop(id).await;
            } else {
                let _ = m.remove(id);
            }
        });
    }

    /// Recompute repository insights from the `.xencode` snapshot (F2-01).
    /// Shared by the AdvisePanel and `/advise` so both always agree. An
    /// un-indexed workspace clears the list and leaves the reason in
    /// `advise_status`.
    pub fn refresh_advise(&mut self) {
        let root = xencode_context_rs::default_root();
        match xencode_context_rs::advise_from_snapshot(&root) {
            Ok(items) => {
                self.advise_items = items;
                self.advise_status.clear();
                self.advise_selected = self
                    .advise_selected
                    .min(self.advise_items.len().saturating_sub(1));
            }
            Err(_) => {
                self.advise_items.clear();
                self.advise_status = "No project index — run /init first.".to_string();
            }
        }
    }

    /// Re-read `git worktree list` for the current directory and mark each
    /// worktree dirty via `dirty_paths`. Failures clear the list and land in
    /// `worktree_status` (not a git repo is the common one).
    pub fn refresh_worktrees(&mut self) {
        let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        match xencode_context_rs::worktree_list(&root) {
            Ok(list) => {
                self.worktree_dirty = list
                    .iter()
                    .map(|w| !xencode_context_rs::dirty_paths(&w.path).is_empty())
                    .collect();
                self.worktrees = list;
                self.worktree_selected = self
                    .worktree_selected
                    .min(self.worktrees.len().saturating_sub(1));
            }
            Err(e) => {
                self.worktrees.clear();
                self.worktree_dirty.clear();
                self.worktree_status = format!("error: {e}");
            }
        }
    }

    /// Runs the queued `git worktree add` (branch field empty = let git
    /// branch off the current HEAD at the directory name).
    pub fn worktree_do_add(&mut self) {
        let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let path = std::path::PathBuf::from(&self.worktree_path_buf);
        let branch =
            (!self.worktree_branch_buf.is_empty()).then_some(self.worktree_branch_buf.as_str());
        self.worktree_prompt = crate::focus::WorktreePrompt::None;
        match xencode_context_rs::worktree_add(&root, &path, branch, false) {
            Ok(list) => {
                self.worktree_status = format!("added worktree {}", path.display());
                self.apply_worktree_list(list);
            }
            Err(e) => self.worktree_status = format!("error: {e}"),
        }
        self.worktree_path_buf.clear();
        self.worktree_branch_buf.clear();
    }

    /// Removes the selected worktree; the main worktree is never removable
    /// and git itself additionally refuses dirty worktrees (no force here).
    pub fn worktree_do_remove(&mut self) {
        self.worktree_prompt = crate::focus::WorktreePrompt::None;
        let Some(wt) = self.worktrees.get(self.worktree_selected).cloned() else {
            return;
        };
        if wt.is_main {
            self.worktree_status = "main worktree is not removable".to_string();
            return;
        }
        let root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        match xencode_context_rs::worktree_remove(&root, &wt.path, false) {
            Ok(list) => {
                self.worktree_status = format!("removed worktree {}", wt.path.display());
                self.apply_worktree_list(list);
            }
            Err(e) => self.worktree_status = format!("error: {e}"),
        }
    }

    fn apply_worktree_list(&mut self, list: Vec<xencode_context_rs::WorktreeInfo>) {
        self.worktree_dirty = list
            .iter()
            .map(|w| !xencode_context_rs::dirty_paths(&w.path).is_empty())
            .collect();
        self.worktrees = list;
        self.worktree_selected = self
            .worktree_selected
            .min(self.worktrees.len().saturating_sub(1));
    }

    /// Proactive warning from the real-time watcher (`[WATCH]<kind>|<path>`).
    /// Kind first: kinds never contain `|`, paths legally can.
    ///
    /// Only files the session is actually reasoning about get a chat warning:
    /// ones pinned via `/ctx track`, `/attach`ed, or open in the editor. The
    /// `FileContextTracker` and the `deps.json` snapshot are re-read from disk
    /// each time so concurrent `/ctx track` and `/init` updates are honored
    /// without threading state into the watcher task. No index yet → the
    /// warning still fires, just without the re-check list.
    fn handle_watch_event(&mut self, body: &str) {
        let Some((kind, path)) = body.split_once('|') else {
            return;
        };
        let root = xencode_context_rs::default_root();
        let xencode = root.join(xencode_context_rs::XENCODE_DIR);
        // Live snapshot (F1-02): update the graph for the changed file before
        // asking it who depends on the change.
        live_refresh_snapshot(&root, kind, path);
        let mut tracker = xencode_context_rs::FileContextTracker::new(&xencode);
        tracker.load_from_disk();
        let tracked: HashSet<String> = tracker.state.files.keys().cloned().collect();
        let graph: Vec<xencode_context_rs::DepEdge> =
            xencode_context_rs::read_json(&xencode_context_rs::deps_json_path(&xencode))
                .unwrap_or_default();
        let affected = xencode_context_rs::affected_dependents(
            &graph,
            &[path],
            xencode_context_rs::AFFECTED_MAX_HOPS,
        );
        let dependents: &[String] = affected.get(path).map(Vec::as_slice).unwrap_or(&[]);
        // Dedup against the last visible warning toast, not chat history (E3-03).
        let last_warning =
            crate::toast::last_of_kind(&self.toasts, crate::toast::ToastKind::Warning);
        if let Some(warning) = watch_warning_for(
            path,
            kind,
            &self.attached_files,
            self.opened_file.as_deref(),
            &tracked,
            dependents,
            last_warning,
        ) {
            crate::toast::push(
                &mut self.toasts,
                warning,
                crate::toast::ToastKind::Warning,
                current_timestamp(),
            );
        }
    }

    /// Connect the Collaboration Hub to a real server: the worker logs in,
    /// joins (or creates) the session, and feeds this event loop `[COLLAB]`
    /// tokens. Nothing here is simulated — if the server is unreachable the
    /// hub says so and goes back to disconnected.
    pub fn start_collab_session(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.collab_session_active {
            return;
        }
        self.collab_session_active = true;
        self.collab_sync_status = "connecting".to_string();
        self.collab_error.clear();
        self.collab_members.clear();
        self.collab_activity_log.clear();
        self.collab_activity_log
            .push(format!("🔌 Connecting to {}...", self.collab_server_url));
        self.collab_worker = Some(crate::collab_client::spawn_collab_worker(
            self.collab_server_url.clone(),
            self.collab_session_id.clone(),
            self.collab_username.clone(),
            tx,
        ));
    }

    /// Stop the client task and mark the hub idle. Safe to call twice; an
    /// already-finished worker's handle aborts cheaply.
    pub fn collab_disconnect(&mut self) {
        if let Some(worker) = self.collab_worker.take() {
            worker.abort();
        }
        if self.collab_session_active {
            self.collab_activity_log.push("🔌 Disconnected".to_string());
        }
        self.collab_session_active = false;
        self.collab_sync_status = "disconnected".to_string();
    }

    /// Hang up (if connected) and dial the same server/session again. The
    /// only reconnect there is — the client never retries by itself.
    pub fn collab_retry(&mut self, tx: mpsc::UnboundedSender<String>) {
        self.collab_disconnect();
        self.start_collab_session(tx);
    }

    /// Tab in the hub: cycle the edited field, entering edit mode with it.
    pub fn collab_cycle_field(&mut self) {
        self.collab_field = self.collab_field.next();
        self.collab_editing = true;
    }

    /// One typed character for the hub form.
    pub fn collab_edit_char(&mut self, c: char) {
        match self.collab_field {
            crate::focus::CollabField::Server => self.collab_server_url.push(c),
            crate::focus::CollabField::Username => self.collab_username.push(c),
            crate::focus::CollabField::Session => self.collab_session_id.push(c),
        }
    }

    pub fn collab_edit_backspace(&mut self) {
        match self.collab_field {
            crate::focus::CollabField::Server => {
                self.collab_server_url.pop();
            }
            crate::focus::CollabField::Username => {
                self.collab_username.pop();
            }
            crate::focus::CollabField::Session => {
                self.collab_session_id.pop();
            }
        }
    }

    /// Apply one `[COLLAB]` token from the client worker (the prefix is
    /// already stripped). Grammar lives in `collab_client::frame_to_tokens`.
    pub fn apply_collab_token(&mut self, body: &str) {
        if let Some(s) = body.strip_prefix("status:") {
            self.collab_sync_status = s.to_string();
            if s == "disconnected" {
                // The worker is finished by definition — it just reported
                // its own end. Drop the handle.
                self.collab_session_active = false;
                self.collab_worker = None;
            }
        } else if let Some(id) = body.strip_prefix("session:") {
            self.collab_session_id = id.to_string();
        } else if let Some(json) = body.strip_prefix("members:") {
            // Complete snapshot from the server; swap the list whole.
            match serde_json::from_str::<Vec<xencode_collaboration_rs::wire::MemberInfo>>(json) {
                Ok(members) => {
                    self.collab_members = members
                        .iter()
                        .map(|m| (m.username.clone(), m.role.clone(), "connected".to_string()))
                        .collect();
                }
                Err(_) => {
                    self.collab_error = "malformed members list from server".to_string();
                }
            }
        } else if let Some(msg) = body.strip_prefix("log:") {
            self.collab_activity_log.push(msg.to_string());
        } else if let Some(msg) = body.strip_prefix("error:") {
            self.collab_error = msg.to_string();
        } else if body == "ready" {
            self.collab_last_sync = current_timestamp();
        }
    }

    /// Panel Enter: run what is typed in the command box.
    pub fn run_bytebot(&mut self, tx: mpsc::UnboundedSender<String>) {
        let task = self.bytebot_command.trim().to_string();
        if task.is_empty() {
            return;
        }
        if let Some(run) = self.arm_bytebot(&task) {
            tokio::spawn(agent_rounds(run, tx));
        }
    }

    /// Start an autonomous ByteBot run (I2-04). This is the ordinary chat tool
    /// loop — same tools, same permission gate, same checkpoints, same plan
    /// strip — with the task as its only user turn. Every step row in the
    /// panel is a call the model actually made and its real outcome, and a run
    /// that fails says so instead of playing a script.
    ///
    /// The caller spawns: arming is state only, so a test can check what the
    /// panel promises without firing a provider request.
    fn arm_bytebot(&mut self, task: &str) -> Option<AgentRun> {
        let task = task.trim();
        if task.is_empty() {
            self.system_line("usage: /bytebot <task>   (or Ctrl+B to open the panel)");
            return None;
        }
        if self.bytebot_running {
            self.push_toast(
                crate::toast::ToastKind::Warning,
                "ByteBot is already working — Ctrl+B to watch it".to_string(),
            );
            return None;
        }
        let task = task.to_string();
        if self.bytebot_history.last().is_none_or(|last| *last != task) {
            self.bytebot_history.push(task.clone());
            if self.bytebot_history.len() > INPUT_HISTORY_LIMIT {
                self.bytebot_history.remove(0);
            }
        }
        self.bytebot_command = task.clone();
        self.bytebot_cursor = task.len();
        self.bytebot_running = true;
        self.bytebot_progress = 0.0;
        self.bytebot_steps.clear();
        self.bytebot_log.clear();
        self.bytebot_log.push(format!("⚡ task: {task}"));
        self.focus = FocusArea::ByteBotPanel;

        let context_messages = self.bytebot_context(&task);
        Some(self.agent_run(LoopSink::ByteBot, context_messages))
    }

    /// ByteBot's conversation: the task framed as an autonomous brief, on the
    /// same live project context a chat turn gets (guidelines, git, retrieval)
    /// and with the tool vocabulary appended. No chat history — a delegated
    /// run starts from the repository, not from whatever was said before.
    fn bytebot_context(&self, task: &str) -> Vec<ChatMessage> {
        self.delegated_context(&xencode_context_rs::default_root(), task, BYTEBOT_BRIEF)
    }

    /// The shared delegated-run prompt builder. `root` is where the run's
    /// tools operate — the main checkout for ByteBot, a fresh worktree for
    /// `/spawn` (I3-03) — so context is collected inside that sandbox.
    fn delegated_context(
        &self,
        root: &std::path::Path,
        task: &str,
        brief: &str,
    ) -> Vec<ChatMessage> {
        let live = xencode_context_rs::collect_live_context(root, task, CTX_PROFILE);
        let model = self.config.default_model.clone();
        let context_window = xencode_providers_rs::capabilities_for(&model).context_window;
        let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
            profile: CTX_PROFILE,
            context_window,
            system: CTX_SYSTEM,
            agents_md: live.agents_md.as_deref(),
            anchor_md: live.anchor_md.as_deref(),
            state_md: live.state_md.as_deref(),
            git_summary: &live.git_summary,
            retrieved: live.blocks,
            attached_block: "",
            history: &[],
            prompt: &format!("{brief}{task}"),
        });
        let mut messages: Vec<ChatMessage> = assembly
            .turns
            .into_iter()
            .map(|t| ChatMessage {
                role: t.role,
                content: t.content.into(),
            })
            .collect();
        if let Some(system) = messages.first_mut().filter(|m| m.role == "system") {
            if let xencode_providers_rs::MessageContent::Text(text) = &mut system.content {
                text.push_str(crate::agent_tools::TOOL_HINT);
            }
        }
        messages
    }

    /// Start a `/spawn <task> [#branch]` subagent (I3-03): create a fresh
    /// git worktree (a sibling of this checkout) and return an `AgentRun`
    /// whose tool sandbox is that worktree. The caller spawns — arming only
    /// mutates state and does the git call, so a test can check what `/spawn
    /// status` promises without firing a provider request.
    fn arm_spawn(&mut self, task: &str, branch: Option<&str>) -> Option<(u64, String, AgentRun)> {
        let task = task.trim();
        if task.is_empty() {
            self.system_line("usage: /spawn <task>   (#branch creates a named worktree)");
            return None;
        }
        let id = self.spawn_next_id;
        let root = xencode_context_rs::default_root();
        let branch_name = branch
            .map(|b| {
                b.strip_prefix('#')
                    .unwrap_or(b)
                    .replace([' ', '/', '\\'], "-")
            })
            .unwrap_or_else(|| format!("xencode/spawn-{id}"));
        let worktree_path = match spawn_worktree(&root, id, &branch_name) {
            Ok(path) => path,
            Err(e) => {
                self.system_line(&format!("⏺ spawn #{id} could not start: {e}"));
                return None;
            }
        };
        self.spawn_next_id += 1;
        self.spawns.push(SpawnRecord {
            id,
            branch: branch_name.clone(),
            path: worktree_path.clone(),
            task: task.to_string(),
            running: true,
            failed: false,
            steps: Vec::new(),
        });
        // The worktree list (Ctrl+O) should show it immediately.
        self.refresh_worktrees();

        // A per-spawn checkpoint store: `/rewind` in the main chat reaches the
        // main checkout's turns, never a spawned worktree's edits.
        let context_messages = self.delegated_context(&worktree_path, task, SPAWN_BRIEF);
        let mut run = self.agent_run(LoopSink::Spawn(id), context_messages);
        run.tool_root = worktree_path;
        run.approval.checkpoints = std::sync::Arc::new(crate::agent_tools::CheckpointStore::new());
        Some((id, branch_name, run))
    }

    /// Resolve the worktree for a `#[branch]`-suffixed `/spawn` task. Pure
    /// parsing, so the git call below can be held to one doc'd rule: the
    /// trailing token, when it starts with `#`, is the branch.
    fn parse_spawn(task_and_branch: &str) -> (&str, Option<&str>) {
        let mut words = task_and_branch.split_whitespace();
        let Some(last) = words.next_back() else {
            return (task_and_branch, None);
        };
        if last.starts_with('#') {
            let task = task_and_branch[..task_and_branch.len() - last.len()].trim();
            (task, Some(last))
        } else {
            (task_and_branch, None)
        }
    }

    /// Apply one `/spawn` loop event. Pure state, tested without a model. A
    /// finished run posts its report into the chat transcript.
    pub fn spawn_event(&mut self, id: u64, body: &str) {
        let Some(i) = self.spawns.iter().position(|s| s.id == id) else {
            return;
        };
        if let Some(summary) = body.strip_prefix("call:") {
            self.spawns[i]
                .steps
                .push((summary.to_string(), "running".to_string()));
            return;
        }
        if let Some(outcome) = body.strip_prefix("done:") {
            if let Some(last) = self.spawns[i].steps.last_mut() {
                last.1 = outcome.to_string();
            }
            return;
        }
        if let Some(text) = body.strip_prefix("log:") {
            // One-line model text while the run is live; not shown on /spawn
            // status, which answers the where/what/whether question.
            let _ = text;
            return;
        }
        if let Some(text) = body.strip_prefix("err:") {
            self.spawns[i].failed = true;
            if let Some(last) = self.spawns[i].steps.last_mut() {
                if last.1 == "running" {
                    last.1 = "failed".to_string();
                }
            }
            self.system_line(&format!("⏺ spawn #{id} failed — {text}"));
            return;
        }
        if let Some(text) = body.strip_prefix("finish:") {
            self.spawns[i].running = false;
            let (id, task, line, final_text) = {
                let rec = &self.spawns[i];
                let line = format!(
                    "⏺ spawn #{id} {} — branch `{}` at `{}`, {}",
                    if rec.failed { "failed" } else { "done" },
                    rec.branch,
                    rec.path.display(),
                    rec.finished_line()
                );
                (rec.id, rec.task.clone(), line, text.to_string())
            };
            self.system_line(&line);
            if !final_text.trim().is_empty() {
                self.messages.push(UiMessage {
                    role: "assistant".to_string(),
                    content: format!("(spawn #{id} · {task})\n{final_text}"),
                });
            }
        }
    }

    /// Handle `/spawn`, `/spawn status` and `/spawn <task> [#branch]`.
    fn handle_spawn_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let arg = prompt
            .strip_prefix("/spawn")
            .unwrap_or("")
            .trim()
            .to_string();
        if arg == "status" {
            if self.spawns.is_empty() {
                self.system_line("No subagents spawned yet — try `/spawn <task>`.");
                return;
            }
            let lines: Vec<String> = self
                .spawns
                .iter()
                .map(|rec| {
                    let state = if rec.running {
                        "running"
                    } else if rec.failed {
                        "failed "
                    } else {
                        "done   "
                    };
                    let task = crate::agent_tools::truncate_one_line(&rec.task, 60);
                    format!(
                        "#{} {} `{}` @ {} · {} · {} call(s)",
                        rec.id,
                        state,
                        rec.branch,
                        rec.path.display(),
                        task,
                        rec.steps.len()
                    )
                })
                .collect();
            for line in lines {
                self.system_line(&line);
            }
            return;
        }
        if arg == "stop" || arg == "abort" {
            self.system_line("No spawn is cancellable mid-run yet — let it finish, or close the worktree with Ctrl+O.");
            return;
        }
        let (task, branch) = Self::parse_spawn(&arg);
        let task = task.trim();
        if task.is_empty() {
            self.system_line("usage: /spawn <task>   (#branch creates a named worktree)");
            return;
        }
        if let Some((id, branch, run)) = self.arm_spawn(task, branch) {
            self.system_line(&format!(
                "⏺ Spawn #{id} started — branch `{branch}` at `{}`",
                run.tool_root.display()
            ));
            tokio::spawn(agent_rounds(run, tx));
        }
    }

    /// Apply one ByteBot loop event. Pure state, so the panel's honesty —
    /// steps are calls, progress is the share of them that finished — is
    /// testable without a model.
    pub fn bytebot_event(&mut self, body: &str) {
        if let Some(summary) = body.strip_prefix("call:") {
            self.bytebot_steps
                .push((summary.to_string(), "running".to_string()));
        } else if let Some(outcome) = body.strip_prefix("done:") {
            if let Some(last) = self.bytebot_steps.last_mut() {
                last.1 = outcome.to_string();
            }
        } else if let Some(text) = body.strip_prefix("err:") {
            self.bytebot_log.push(format!("❌ {text}"));
            if let Some(last) = self.bytebot_steps.last_mut() {
                if last.1 == "running" {
                    last.1 = "failed".to_string();
                }
            }
        } else if let Some(text) = body.strip_prefix("log:") {
            self.bytebot_log.push(text.to_string());
        }
        self.bytebot_progress = bytebot_progress(&self.bytebot_steps);
    }

    /// Handle `/init`, `/init abort` and `/init status` chat commands.
    fn handle_init_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let command = prompt.strip_prefix("/init").unwrap_or("").trim();
        match command {
            "abort" => {
                if self.init_running {
                    self.init_cancel.store(true, Ordering::Relaxed);
                    let _ = tx.send(
                        "[INIT]log:⏹️ Abort requested — finishing the current step…".to_string(),
                    );
                } else {
                    let _ = tx.send("[INIT]log:ℹ️ No /init job is running.".to_string());
                }
            }
            "status" => {
                let done = self.init_steps.iter().filter(|(_, s)| s == "done").count();
                let line = if self.init_running {
                    format!(
                        "⏳ init running — {done}/{} steps, {}%. Use /init abort to stop.",
                        self.init_steps.len(),
                        (self.init_progress * 100.0) as u64
                    )
                } else if self.init_visible || !self.init_log.is_empty() {
                    format!(
                        "🗂  Last /init run: {} step(s), {} log line(s). Type /init to re-run.",
                        done,
                        self.init_log.len()
                    )
                } else {
                    "No project index built yet — type /init to scan and create it.".to_string()
                };
                let _ = tx.send(format!("[INIT]log:{}", line));
            }
            "" => {
                if self.init_running {
                    let _ = tx.send(
                        "[INIT]log:⚠️ An init job is already running — use /init abort to stop it."
                            .to_string(),
                    );
                } else {
                    self.init_cancel.store(false, Ordering::Relaxed);
                    self.run_project_init(tx);
                }
            }
            other => {
                let _ = tx.send(format!(
                    "[INIT]log:ℹ️ Unknown /init subcommand '{other}' — use /init, /init abort, or /init status."
                ));
            }
        }
    }

    /// Run the deterministic structural `/init` pass in the background.
    /// Progress and log lines stream back through `[INIT]` channel tokens.
    pub fn run_project_init(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.init_running {
            return;
        }
        const PHASES: [&str; 7] = [
            "Create .xencode directory",
            "Resume check",
            "Git snapshot",
            "Scan repository",
            "Analyze languages & sizes",
            "Extract symbols & dependencies",
            "Write index files",
        ];

        self.init_running = true;
        self.init_progress = 0.0;
        self.init_visible = true;
        self.init_steps = PHASES
            .iter()
            .map(|name| (name.to_string(), "pending".to_string()))
            .collect();
        self.init_log.clear();
        self.init_log.push(
            "⏺️ Initializing project context (structural pass) — zero LLM calls.".to_string(),
        );

        let cancel = self.init_cancel.clone();
        let root = xencode_context_rs::default_root();

        tokio::spawn(async move {
            let tx_progress = tx.clone();
            let progress = move |line: &str| {
                if let Some(name) = line.strip_prefix("phase_start:") {
                    if let Some(idx) = PHASES.iter().position(|p| *p == name) {
                        let _ = tx_progress.send(format!("[INIT]step:{idx}:running:{name}"));
                        let _ = tx_progress.send(format!(
                            "[INIT]progress:{:.2}",
                            idx as f64 / PHASES.len() as f64
                        ));
                    }
                } else if let Some(name) = line.strip_prefix("phase_done:") {
                    if let Some(idx) = PHASES.iter().position(|p| *p == name) {
                        let _ = tx_progress.send(format!("[INIT]step:{idx}:done:{name}"));
                        let _ = tx_progress.send(format!(
                            "[INIT]progress:{:.2}",
                            (idx as f64 + 1.0) / PHASES.len() as f64
                        ));
                    }
                } else if let Some(msg) = line.strip_prefix("log:") {
                    let _ = tx_progress.send(format!("[INIT]log:{msg}"));
                }
            };

            let result =
                tokio::task::spawn_blocking(move || init_project(&root, cancel, progress)).await;

            match result {
                Ok(Ok(summary)) => {
                    if summary.fresh {
                        let _ = tx.send(
                            "[INIT]log:✅ Project context is already up to date — nothing rewritten."
                                .to_string(),
                        );
                    } else {
                        let skipped = if summary.skipped > 0 {
                            format!(" ({} ignored/excluded)", summary.skipped)
                        } else {
                            String::new()
                        };
                        let _ = tx.send(format!(
                            "[INIT]log:✅ Indexed {} files{skipped} across {} language(s) — {} LOC.",
                            summary.files_scanned,
                            summary.languages.len(),
                            summary.total_loc
                        ));
                        if summary.dep_edges > 0 || summary.symbol_files > 0 {
                            let _ = tx.send(format!(
                                "[INIT]log:🧩 {} file(s) with symbols · {} dependency edge(s)",
                                summary.symbol_files, summary.dep_edges
                            ));
                        }
                        if !summary.secret_files.is_empty() {
                            let _ = tx.send(format!(
                                "[INIT]log:🔒 {} secret-detected file(s) listed, not read.",
                                summary.secret_files.len()
                            ));
                        }
                        if !summary.binary_files.is_empty() {
                            let _ = tx.send(format!(
                                "[INIT]log:🧊 {} binary file(s) listed, not read.",
                                summary.binary_files.len()
                            ));
                        }
                        if let Some(g) = &summary.git {
                            let head = if g.head.len() > 8 {
                                g.head[..8].to_string()
                            } else {
                                g.head.clone()
                            };
                            let _ = tx.send(format!(
                                "[INIT]log:🎋 {} @ {head} — {} dirty file(s)",
                                g.branch, g.dirty
                            ));
                        }
                        let _ = tx.send(format!(
                            "[INIT]log:🗂  index + symbols + deps — {} bytes",
                            summary.index_bytes
                        ));
                    }
                    let _ = tx.send(
                        "[INIT]log:💡 Index refreshes automatically on git changes. /init status shows the last run."
                            .to_string(),
                    );
                }
                Ok(Err(err)) => {
                    let _ = tx.send(format!("[INIT]log:❌ init failed: {err}"));
                }
                Err(join_err) => {
                    let _ = tx.send(format!("[INIT]log:❌ init task panicked: {join_err}"));
                }
            }
            let _ = tx.send("[INIT_DONE]".to_string());
        });
    }

    /// Handle `/ctx` — context assembly: deterministic retrieval over the
    /// project index, stale-file status, and pinning a file as "loaded".
    fn handle_ctx_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let rest = prompt.strip_prefix("/ctx").unwrap_or("").trim();
        let mut parts = rest.split_whitespace();
        match parts.next() {
            Some("status") => {
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let mut tracker = xencode_context_rs::FileContextTracker::new(&xencode);
                tracker.load_from_disk();
                let _ = tx.send("[CTX_START]".to_string());
                if tracker.state.is_empty() {
                    let _ = tx.send(
                        "[CTX]ℹ️ No tracked files — pin one with /ctx track src/foo.rs."
                            .to_string(),
                    );
                    return;
                }
                let all = tracker.check_all(&root);
                let stale: usize = all
                    .iter()
                    .filter(|t| t.state == xencode_context_rs::FileStateKind::Stale)
                    .count();
                let missing: usize = all
                    .iter()
                    .filter(|t| t.state == xencode_context_rs::FileStateKind::Missing)
                    .count();
                for t in &all {
                    let mark = match t.state {
                        xencode_context_rs::FileStateKind::Clean => "✔",
                        xencode_context_rs::FileStateKind::Stale => "⚠",
                        xencode_context_rs::FileStateKind::Missing => "✖",
                    };
                    let _ = tx.send(format!("[CTX]{mark} {} — {:?}", t.path, t.state));
                }
                let _ = tx.send(format!(
                    "[CTX]📊 {} tracked — {stale} stale, {missing} missing.",
                    all.len()
                ));
            }
            Some("track") => {
                let Some(path) = parts.next() else {
                    let _ = tx.send("[CTX_START]".to_string());
                    let _ = tx.send("[CTX]ℹ️ Usage: /ctx track <repo-relative-path>".to_string());
                    return;
                };
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let mut tracker = xencode_context_rs::FileContextTracker::new(&xencode);
                tracker.load_from_disk();
                tracker.mark_loaded(&root, &[path]);
                let _ = tx.send("[CTX_START]".to_string());
                if tracker.save().is_ok() {
                    let _ = tx.send(format!(
                        "[CTX]🔖 Pinned \"{path}\" at load-time hash — /ctx status to check staleness."
                    ));
                } else {
                    let _ = tx.send("[CTX]❌ Could not persist the tracking state.".to_string());
                }
            }
            Some("compact") => {
                let (mut t, appended) = self.canonical_transcript();
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let path = xencode_context_rs::Transcript::current_path(&xencode);
                let snap = t.snapshot(&xencode).unwrap_or_default();
                let report = xencode_context_rs::soft_compact(&mut t, 0.70);
                let _ = t.save_to(&path);
                self.messages = t
                    .entries
                    .iter()
                    .map(|e| UiMessage {
                        role: e.role.clone(),
                        content: e.content.clone(),
                    })
                    .collect();
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]📚 Canonical transcript synced (+{appended} new) → {} entries",
                    t.entries.len()
                ));
                let _ = tx.send(format!(
                    "[CTX]🗜️ Soft compaction {} → {} entries (dropped {}), decisions kept: {}",
                    report.before, report.after, report.dropped, report.retained_decisions
                ));
                let _ = tx.send(format!("[CTX]💾 Pre-rewrite snapshot → {}", snap.display()));
                let _ = tx.send(
                    "[CTX]✅ Deterministic, no LLM call — state.md only changes when the model flags it. Chat now shows the working projection."
                        .to_string(),
                );
            }
            Some("eval") => {
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let Some(index) = xencode_context_rs::RetrievalIndex::load(&xencode) else {
                    let _ = tx.send("[CTX_START]".to_string());
                    let _ = tx.send("[CTX]❌ No project index — run /init first.".to_string());
                    return;
                };
                let gold = xencode_context_rs::gold_from_disk(&xencode);
                let dirty: HashSet<String> =
                    xencode_context_rs::dirty_paths(&root).into_iter().collect();
                let k = 5;
                let base = xencode_context_rs::evaluate(&index, &gold, k, &dirty, false);
                let reranked = xencode_context_rs::evaluate(&index, &gold, k, &dirty, true);
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]🧪 Retrieval eval — {} gold queries, top-{} ({} gold file{})",
                    base.queries,
                    k,
                    if gold.iter().all(|g| g.expected.is_empty()) {
                        0
                    } else {
                        base.queries
                    },
                    if base.queries == 1 { "" } else { "s" }
                ));
                let _ = tx.send(format!(
                    "[CTX]   deterministic : MRR {:.3} · recall@1 {:.0}% · recall@3 {:.0}% · P@1 {:.0}%",
                    base.mrr,
                    base.recall_at.first().map(|v| v * 100.0).unwrap_or(0.0),
                    base.recall_at.get(2).map(|v| v * 100.0).unwrap_or(0.0),
                    base.precision_at.first().map(|v| v * 100.0).unwrap_or(0.0),
                ));
                let _ = tx.send(format!(
                    "[CTX]   + BM25 rerank : MRR {:.3} · recall@1 {:.0}% · recall@3 {:.0}% · P@1 {:.0}%",
                    reranked.mrr,
                    reranked.recall_at.first().map(|v| v * 100.0).unwrap_or(0.0),
                    reranked.recall_at.get(2).map(|v| v * 100.0).unwrap_or(0.0),
                    reranked.precision_at.first().map(|v| v * 100.0).unwrap_or(0.0),
                ));
                let delta = reranked.mrr - base.mrr;
                let verdict = if (delta - base.mrr).abs() < f64::EPSILON && delta.abs() < 1e-6 {
                    "no change"
                } else if delta > 1e-6 {
                    "rerank wins — enable by default"
                } else {
                    "rerank ties or hurts — keep deterministic baseline"
                };
                let _ = tx.send(format!("[CTX]   ΔMRR {delta:+.3} → {verdict}",));
                let _ = tx.send("[CTX]   Per query:".to_string());
                for (query, expected, rank, ranked) in &base.hits {
                    let rank_str = if *rank == usize::MAX {
                        "miss".to_string()
                    } else {
                        format!("#{}", rank)
                    };
                    let _ = tx.send(format!(
                        "[CTX]     {rank_str:>5}  {query}  → {}",
                        expected.join(", ")
                    ));
                    if *rank == usize::MAX {
                        let _ = tx.send(format!(
                            "[CTX]          top: {}",
                            ranked
                                .iter()
                                .take(3)
                                .cloned()
                                .collect::<Vec<String>>()
                                .join(" · ")
                        ));
                    }
                }
            }
            Some("kv") => {
                const PROFILE: xencode_context_rs::HardwareProfile =
                    xencode_context_rs::HardwareProfile::Balanced;
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
                let anchor = std::fs::read_to_string(xencode.join("anchor.md")).ok();
                let state =
                    xencode_context_rs::ContextState::from_disk(&xencode).map(|s| s.to_markdown());
                let git = xencode_context_rs::git_summary_text(&root).unwrap_or_default();
                let recent_a = "user: how does auth work?\nassistant: it uses the auth module";
                let recent_b = "user: why is startup slow?\nassistant: profile the init path";
                // Different recent windows (and git text) must NOT disturb the
                // byte-stable head — that's the KV-reuse contract (§13).
                let doc_a = xencode_context_rs::assemble_prompt(
                    PROFILE,
                    CTX_SYSTEM,
                    agents.as_deref(),
                    anchor.as_deref(),
                    state.as_deref(),
                    &git,
                    Vec::new(),
                    recent_a,
                );
                let doc_b = xencode_context_rs::assemble_prompt(
                    PROFILE,
                    CTX_SYSTEM,
                    agents.as_deref(),
                    anchor.as_deref(),
                    state.as_deref(),
                    &git,
                    Vec::new(),
                    recent_b,
                );
                let stable_ok = doc_a.stable_prefix == doc_b.stable_prefix;
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]🗂 Profile {} — ctx {} · utilization {}% · top-k {}",
                    PROFILE.name(),
                    PROFILE.ctx_tokens(),
                    (PROFILE.utilization() * 100.0) as u64,
                    PROFILE.top_k(),
                ));
                let _ = tx.send(format!(
                    "[CTX]⚙️ llama.cpp args: {}",
                    PROFILE.llama_cpp_args().join(" ")
                ));
                let _ = tx.send(format!(
                    "[CTX]🧱 Stable prefix {} bytes — sha256 {} · cross-request identical: {}",
                    doc_a.stable_prefix.len(),
                    doc_a.stable_prefix_sha256(),
                    if stable_ok {
                        "✅ yes"
                    } else {
                        "❌ NO — KV reuse is broken"
                    }
                ));

                let rows = xencode_context_rs::read_metrics(&xencode);
                if rows.is_empty() {
                    let _ = tx.send("[CTX]📈 No metrics yet — run /ctx <query> then a llama.cpp generation to see KV reuse.".to_string());
                } else {
                    let _ = tx.send("[CTX]📈 Latest KV-cache rows per profile:".to_string());
                    for r in xencode_context_rs::RequestMetrics::latest_per_profile(&rows) {
                        let _ = tx.send(format!(
                            "[CTX]   {} — prompt {} · cached {} · reuse {}% · {} tok/s",
                            r.profile,
                            r.prompt_tokens,
                            r.cached_tokens,
                            (r.kv_reuse_ratio() * 100.0) as u64,
                            r.generation_tok_s
                        ));
                    }
                    // Interpret §13: large stable prefix + ~0 cached = prefix drift bug.
                    let latest = xencode_context_rs::RequestMetrics::latest_per_profile(&rows)
                        .first()
                        .cloned();
                    if let Some(r) = latest {
                        if r.prompt_tokens > 2000 && r.kv_reuse_ratio() < 0.05 {
                            let _ = tx.send(
                                "[CTX]🚨 Large prompt but ~0 cached tokens — something breaks prefix stability; check for dynamic tiers above the stable head."
                                    .to_string(),
                            );
                        }
                    }
                }
                if let Some(ts) = &self.last_llamacpp_timings {
                    let _ = tx.send(format!(
                        "[CTX]⚡ Last llama.cpp run — evaluated {} · generated {} · {} tok/s gen · {} tok/s prompt",
                        ts.tokens_evaluated,
                        ts.tokens_generated,
                        (ts.predicted_per_second as u64),
                        (ts.prompt_per_second as u64)
                    ));
                }
            }
            Some("archive") => {
                let (t, appended) = self.canonical_transcript();
                let root = xencode_context_rs::default_root();
                let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                let state =
                    xencode_context_rs::ContextState::from_disk(&xencode).unwrap_or_default();
                let snap = t.snapshot(&xencode).unwrap_or_default();
                let prompt = xencode_context_rs::hard_compact_prompt(&state, &t);
                let _ = tx.send("[CTX_START]".to_string());
                let _ = tx.send(format!(
                    "[CTX]📚 Canonical transcript synced (+{appended} new) → {} entries",
                    t.entries.len()
                ));
                let _ = tx.send(format!(
                    "[CTX]💾 Archived snapshot → {} — raw history is safe.",
                    snap.display()
                ));
                let _ = tx.send(format!(
                    "[CTX]🧠 Hard-compaction fold prompt ({} tokens):",
                    xencode_context_rs::est_tokens(prompt.len(), true)
                ));
                for line in prompt.lines() {
                    let _ = tx.send(format!("[CTX]    {line}"));
                }
            }
            _ => {
                let query = if let Some(rest) = rest.strip_prefix("retrieve") {
                    rest.trim().to_string()
                } else {
                    rest.to_string()
                };
                // Carry a small recent-message window for the assembly preview.
                let mut recent: Vec<String> = self
                    .messages
                    .iter()
                    .rev()
                    .take(8)
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect();
                recent.reverse();
                self.run_ctx_retrieval(query, recent.join("\n"), tx);
            }
        }
    }

    /// Repository insights (`/advise [filter]`) — deterministic refactor
    /// suggestions + bug warnings from the live `.xencode` snapshot
    /// ([`Self::refresh_advise`]), streamed back through `[ADVISE]` chat
    /// lines. An optional substring narrows the report to matching files
    /// (e.g. `/advise router`).
    fn handle_advise_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let filter = prompt.strip_prefix("/advise").unwrap_or("").trim();
        let filter = if filter.is_empty() {
            None
        } else {
            Some(filter)
        };
        self.refresh_advise();
        let _ = tx.send("[ADVISE_START]".to_string());
        if !self.advise_status.is_empty() {
            let _ = tx.send(format!("[ADVISE]❌ {}", self.advise_status));
            return;
        }
        for line in format_advise_report(&self.advise_items, filter) {
            let _ = tx.send(format!("[ADVISE]{line}"));
        }
    }

    /// `/rewind [turns]` — put back the files the agent changed in its most
    /// recent turns that touched anything (default: the last turn). The
    /// snapshots are session-only bytes in memory, so this can never undo an
    /// earlier xencode run, and git is left entirely alone.
    fn handle_rewind_command(&mut self, prompt: &str) {
        // ByteBot writes through the same gate, so rewinding under it would
        // fight a run that is still going.
        if self.is_generating || self.bytebot_running {
            self.push_toast(
                crate::toast::ToastKind::Warning,
                "can't rewind while the agent is working — Esc to stop it first".to_string(),
            );
            return;
        }
        let arg = prompt.strip_prefix("/rewind").unwrap_or("").trim();
        let back = if arg.is_empty() {
            1
        } else {
            match arg.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => {
                    self.system_line("usage: /rewind [turns] — a whole number of turns, default 1");
                    return;
                }
            }
        };
        let available = self.checkpoints.turns();
        if available == 0 {
            self.system_line(
                "Nothing to rewind: the agent has not changed any files in this session.",
            );
            return;
        }
        let report = self.checkpoints.rewind(back.min(available));
        self.refresh_editor_after_rewind(&report);
        let restored = report.files.len() - report.removed;
        let mut parts: Vec<String> = Vec::new();
        if restored > 0 {
            parts.push(format!("{restored} put back"));
        }
        if report.removed > 0 {
            parts.push(format!("{} deleted", report.removed));
        }
        if !report.failed.is_empty() {
            parts.push(format!("{} failed", report.failed.len()));
        }
        self.system_line(&format!(
            "↺ Rewound {} agent turn(s) — {} ({})",
            report.turns,
            parts.join(", "),
            if report.files.len() > 4 {
                format!(
                    "{}, …",
                    report
                        .files
                        .iter()
                        .take(4)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                report.files.join(", ")
            }
        ));
        self.push_toast(
            crate::toast::ToastKind::Info,
            format!("rewound {} file(s)", report.files.len()),
        );
    }

    /// `/plan` toggles the agent's todo strip between its compact form (the
    /// first few steps, always visible while a plan exists) and the full list;
    /// `/plan clear` drops the list the model posted without asking it to.
    fn handle_plan_command(&mut self, prompt: &str) {
        let arg = prompt.strip_prefix("/plan").unwrap_or("").trim();
        let items = crate::agent_tools::plan_items(&self.agent_plan);
        match arg {
            "clear" => {
                if items.is_empty() {
                    self.system_line("There is no plan to clear.");
                    return;
                }
                if let Ok(mut plan) = self.agent_plan.lock() {
                    plan.clear();
                }
                self.plan_pinned = false;
                self.system_line("Plan cleared. The agent can post a new one.");
            }
            "" => {
                if items.is_empty() {
                    self.system_line(
                        "No plan yet. Ask the agent to plan the work and it will post one here.",
                    );
                    return;
                }
                self.plan_pinned = !self.plan_pinned;
                let done = items
                    .iter()
                    .filter(|item| item.status == crate::agent_tools::PlanStatus::Done)
                    .count();
                self.system_line(&format!(
                    "Plan {}: {done}/{} steps done{}",
                    if self.plan_pinned {
                        "pinned"
                    } else {
                        "compact"
                    },
                    items.len(),
                    if self.plan_pinned || items.len() <= crate::agent_tools::PLAN_COMPACT_ITEMS {
                        String::new()
                    } else {
                        format!(" — /plan again to see all {}", items.len())
                    }
                ));
            }
            _ => self.system_line("usage: /plan (toggle the full list)  |  /plan clear"),
        }
    }

    /// I3-01: the Model Context Protocol, tools only. `/mcp` connects every
    /// server declared under `mcp_servers` in config.json and offers its tools
    /// to the model for the session; `/mcp status` lists what is running and
    /// what non-protocol noise it has printed; `/mcp stop` kills everything and
    /// withdraws the tools. A configured-but-broken server must not stall TUI
    /// startup, so nothing here is started unless the user asks.
    fn handle_mcp_command(&mut self, prompt: &str, tx: mpsc::UnboundedSender<String>) {
        let arg = prompt.strip_prefix("/mcp").unwrap_or("").trim();
        match arg {
            "status" => {
                let mcp = self.mcp.clone();
                tokio::spawn(async move {
                    for line in mcp.status_lines().await {
                        let _ = tx.send(format!("[MCP]{line}"));
                    }
                });
            }
            "stop" => {
                let mcp = self.mcp.clone();
                tokio::spawn(async move {
                    let stopped = mcp.stop_all().await;
                    let _ = tx.send(
                        "[MCP]".to_owned()
                            + &if stopped == 0 {
                                "no MCP servers running.".to_string()
                            } else {
                                format!("stopped {stopped} MCP server(s) and withdrew their tools.")
                            },
                    );
                });
            }
            "" => {
                if self.config.mcp_servers.is_empty() {
                    self.system_line(
                        "No MCP servers configured — add a \"mcp_servers\" block to config.json, then run /mcp.",
                    );
                    return;
                }
                let specs: Vec<crate::mcp::ServerSpec> = self
                    .config
                    .mcp_servers
                    .iter()
                    .map(|(name, server)| crate::mcp::spec_from_config(name, server))
                    .collect();
                self.system_line(&format!("Connecting {} MCP server(s)…", specs.len()));
                let mcp = self.mcp.clone();
                let timeout = std::time::Duration::from_secs(self.config.mcp_timeout.max(1));
                tokio::spawn(async move {
                    for report in mcp.connect(&specs, timeout).await {
                        let _ = tx.send(if report.connected {
                            format!("[MCP]✓ {} · {}", report.server, report.detail)
                        } else {
                            format!("[MCP]✗ {} · {}", report.server, report.detail)
                        });
                    }
                });
            }
            _ => self.system_line("usage: /mcp (connect all)  ·  /mcp status  ·  /mcp stop"),
        }
    }

    /// A rewind that touches the file the user is looking at must not leave
    /// a stale buffer in the editor — but unsaved edits are the user's, so
    /// those are never silently thrown away.
    fn refresh_editor_after_rewind(&mut self, report: &crate::agent_tools::RewindReport) {
        let Some(opened) = self.opened_file.clone() else {
            return;
        };
        if !report
            .paths
            .iter()
            .any(|path| path == std::path::Path::new(&opened))
        {
            return;
        }
        if self.editor_dirty {
            self.system_line(&format!(
                "⚠ {opened} was rewound but your unsaved editor changes were kept — save would overwrite the rewind."
            ));
            return;
        }
        if std::path::Path::new(&opened).exists() {
            self.open_file_in_editor(&opened);
        } else {
            self.editor = TextArea::new(vec![format!("(rewound: {opened} was deleted)")]);
            self.opened_file = None;
        }
    }

    /// One line in the transcript under the system role (local commands that
    /// never round-trip a provider).
    fn system_line(&mut self, text: &str) {
        self.messages.push(UiMessage {
            role: "system".to_string(),
            content: text.to_string(),
        });
    }

    /// Sync the in-memory conversation into the canonical transcript store.    /// Appends only messages that aren't already at the tail, so repeated
    /// `/ctx compact|archive` runs never double-count history.
    fn canonical_transcript(&mut self) -> (xencode_context_rs::Transcript, usize) {
        let root = xencode_context_rs::default_root();
        let xencode = root.join(xencode_context_rs::XENCODE_DIR);
        let path = xencode_context_rs::Transcript::current_path(&xencode);
        let mut t = xencode_context_rs::Transcript::from_disk(&path)
            .unwrap_or_else(|| xencode_context_rs::Transcript::new("tui"));
        let mem = self.memory.get_context(100_000);
        let mut appended = 0usize;
        for m in &mem {
            let dup = t
                .entries
                .last()
                .map(|e| e.role == m.role && e.content == m.content)
                .unwrap_or(false);
            if dup {
                continue;
            }
            t.add(&m.role, &m.content);
            appended += 1;
        }
        (t, appended)
    }

    /// Append one §13 metrics row for an assembled context. Shared by the
    /// `/ctx` preview and real generations so both report comparable numbers.
    fn record_ctx_metrics(
        xencode: &std::path::Path,
        profile: HardwareProfile,
        total_tokens: u64,
        target_tokens: u64,
        retrieved_included: usize,
        soft_compaction_needed: bool,
    ) {
        let mut m =
            xencode_context_rs::RequestMetrics::new(profile.name(), profile.ctx_tokens() as u32);
        m.ts_unix_ms = (current_timestamp() * 1000.0) as u64;
        m.prompt_tokens = total_tokens.min(u32::MAX as u64) as u32;
        m.retrieved_files = retrieved_included.min(u8::MAX as usize) as u8;
        m.context_usage = (total_tokens as f32 / target_tokens.max(1) as f32).min(1.0);
        m.compaction = if soft_compaction_needed {
            xencode_context_rs::CompactAction::Soft
        } else {
            xencode_context_rs::CompactAction::None
        };
        let _ = xencode_context_rs::append_metrics(xencode, &m);
    }

    /// Run deterministic retrieval + a context-assembly preview in the
    /// background, streaming results back through `[CTX]` chat lines.
    fn run_ctx_retrieval(
        &mut self,
        query: String,
        recent_text: String,
        tx: mpsc::UnboundedSender<String>,
    ) {
        tokio::spawn(async move {
            let _ = tx.send("[CTX_START]".to_string());
            let root = xencode_context_rs::default_root();
            let xencode = root.join(xencode_context_rs::XENCODE_DIR);
            let Some(index) = xencode_context_rs::RetrievalIndex::load(&xencode) else {
                let _ = tx.send("[CTX]❌ No project index — run /init first.".to_string());
                return;
            };
            let profile = xencode_context_rs::HardwareProfile::Balanced;
            let opts = xencode_context_rs::RetrieveOptions {
                top_k: profile.top_k(),
                ..Default::default()
            };
            let changed: HashSet<String> =
                xencode_context_rs::dirty_paths(&root).into_iter().collect();
            let results = xencode_context_rs::retrieve(&query, &index, &changed, &opts);
            if results.is_empty() {
                let _ = tx.send(
                    "[CTX]😶 Nothing above the score threshold — try a more specific query."
                        .to_string(),
                );
                return;
            }
            let _ = tx.send(format!(
                "[CTX]🎯 Retrieval ({} profile, top-{}):",
                profile.name(),
                results.len()
            ));
            for r in &results {
                let _ = tx.send(format!(
                    "[CTX]  {:>3}  {}  ·  {}",
                    r.score,
                    r.path,
                    r.reasons.join(", ")
                ));
            }

            let blocks = xencode_context_rs::read_retrieved_bodies(
                &root,
                &index.files,
                &results,
                profile.content_cap_chars(),
            );
            let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
            let anchor = std::fs::read_to_string(xencode.join("anchor.md")).ok();
            let state = std::fs::read_to_string(xencode.join("state.md")).ok();
            let git = xencode_context_rs::git_summary_text(&root).unwrap_or_default();
            let doc = xencode_context_rs::assemble_prompt(
                profile,
                CTX_SYSTEM,
                agents.as_deref(),
                anchor.as_deref(),
                state.as_deref(),
                &git,
                blocks,
                &recent_text,
            );
            let stable_tokens: u64 = doc.tiers.iter().take(3).map(|t| t.tokens).sum();
            let _ = tx.send(format!(
                "[CTX]📦 Assembled context ≈ {} / {} tokens target — {} / {} retrieved files in — stable prefix {} tokens",
                doc.total_tokens,
                doc.target_tokens,
                doc.retrieved_included,
                doc.retrieved_total,
                stable_tokens
            ));
            // Capture for the KV-reuse metrics on the next llama.cpp timings.
            let _ = tx.send(format!(
                "[CTXSTATS]{}|{}",
                doc.total_tokens.min(u32::MAX as u64),
                doc.retrieved_included.min(u8::MAX as usize)
            ));
            if doc.truncated {
                let _ =
                    tx.send("[CTX]⚠ Some retrieved files dropped to fit the budget.".to_string());
            }
            if doc.soft_compaction_needed {
                let _ = tx.send(
                    "[CTX]⚠ Recent-message budget under 1 message — soft compaction should trigger before sending."
                        .to_string(),
                );
            }

            Self::record_ctx_metrics(
                &xencode,
                profile,
                doc.total_tokens,
                doc.target_tokens,
                doc.retrieved_included,
                doc.soft_compaction_needed,
            );
        });
    }

    /// Record a clip from the microphone (J-07). Enter starts a capture; Enter
    /// again ends it early. Levels come from RMS of the bytes the recorder
    /// actually sent, and text appears only if a whisper CLI is installed —
    /// otherwise the panel keeps the clip and says why there is no text.
    pub fn start_voice_session(
        &mut self,
        root: std::path::PathBuf,
        tx: mpsc::UnboundedSender<String>,
    ) {
        if self.voice_busy {
            return;
        }
        self.voice_active = true;
        self.voice_busy = true;
        self.voice_status = "listening".to_string();
        self.voice_level = 0.0;
        self.voice_peak = 0.0;
        self.voice_pcm_bytes = 0;
        self.voice_note.clear();
        self.voice_clip = None;
        self.voice_recorder.clear();
        self.voice_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.voice_mute_flag
            .store(self.voice_muted, std::sync::atomic::Ordering::Relaxed);

        let Some(recorder) = crate::voice::find_recorder() else {
            self.voice_note =
                "No recorder on PATH — looked for arecord, pw-record, parec. Nothing was captured."
                    .to_string();
            self.voice_status = "idle".to_string();
            self.voice_busy = false;
            return;
        };
        self.voice_recorder = recorder
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| recorder.display().to_string());
        let clip_dir = root.join(".xencode").join("voice");
        if let Err(e) = std::fs::create_dir_all(&clip_dir) {
            self.voice_note = format!("Cannot write clips to {}: {e}", clip_dir.display());
            self.voice_status = "idle".to_string();
            self.voice_busy = false;
            return;
        }

        let stop = self.voice_stop.clone();
        let muted = self.voice_mute_flag.clone();
        let args = crate::voice::recorder_args_for(&recorder);
        tokio::spawn(async move {
            let recorder_tx = tx.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                run_voice_capture(&recorder, &args, &clip_dir, &stop, &muted, &recorder_tx)
            })
            .await;
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let _ = tx.send(format!("[VOICE]err:clip could not be saved: {e}"));
                }
                Err(e) => {
                    let _ = tx.send(format!("[VOICE]err:capture thread died: {e}"));
                }
            }
            let _ = tx.send("[VOICE]status:idle".to_string());
        });
    }

    /// End the current capture early; the clip recorded so far is kept.
    pub fn stop_voice_session(&self) {
        if self.voice_busy {
            self.voice_stop
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn toggle_voice_session(
        &mut self,
        root: std::path::PathBuf,
        tx: mpsc::UnboundedSender<String>,
    ) {
        if self.voice_busy {
            self.stop_voice_session();
        } else {
            self.start_voice_session(root, tx);
        }
    }

    /// Mute is a real switch: the capture thread reads it and discards audio
    /// instead of keeping it, so muting does not produce a silent clip later.
    pub fn set_voice_muted(&mut self, muted: bool) {
        self.voice_muted = muted;
        self.voice_mute_flag
            .store(muted, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn voice_apply_level(&mut self, body: &str) {
        let Some((level, bytes)) = parse_voice_level(body) else {
            return;
        };
        self.voice_level = level;
        self.voice_pcm_bytes = bytes;
        if level > self.voice_peak {
            self.voice_peak = level;
        }
    }

    pub fn voice_apply_clip(&mut self, body: &str) {
        let Some((path, ms)) = body.split_once('|') else {
            self.voice_note = format!("Malformed clip report: {body}");
            return;
        };
        let Ok(ms) = ms.trim().parse::<u64>() else {
            self.voice_note = format!("Malformed clip length in: {body}");
            return;
        };
        self.voice_clip = Some(std::path::PathBuf::from(path));
        self.voice_note = format!("Kept {} of audio at {}.", crate::voice::format_ms(ms), path);
    }

    pub fn voice_apply_transcript(&mut self, text: &str) {
        self.voice_transcript.push(text.to_string());
        self.voice_note.clear();
    }

    pub fn voice_apply_note(&mut self, note: &str) {
        self.voice_note = note.to_string();
        self.voice_level = 0.0;
    }

    pub fn voice_apply_error(&mut self, err: &str) {
        self.voice_note = err.to_string();
        self.voice_status = "idle".to_string();
        self.voice_busy = false;
        self.voice_level = 0.0;
    }

    /// A capture has ended. The panel keeps showing the clip and whatever the
    /// transcriber (or its absence) reported; only the meter goes quiet.
    pub fn voice_finish(&mut self) {
        self.voice_busy = false;
        self.voice_status = "idle".to_string();
        self.voice_level = 0.0;
    }

    pub fn term_asst_char(&mut self, c: char) {
        self.term_asst_query.push(c);
    }

    pub fn term_asst_backspace(&mut self) {
        self.term_asst_query.pop();
    }

    /// Indices into `term_asst_suggestions` that the risk filter lets through,
    /// so selection and rendering agree about what is on screen.
    pub fn term_visible_rows(&self) -> Vec<usize> {
        (0..self.term_asst_suggestions.len())
            .filter(|&i| {
                self.term_risk_filter == "All"
                    || self.term_asst_suggestions[i]
                        .1
                        .eq_ignore_ascii_case(&self.term_risk_filter)
            })
            .collect()
    }

    /// Ask the model once for commands that do what the query says. A reply
    /// that is not a command list is shown as the reply, not converted into
    /// invented suggestions.
    pub fn ask_terminal(&mut self, tx: mpsc::UnboundedSender<String>) {
        let query = self.term_asst_query.trim().to_string();
        if query.is_empty() {
            self.term_asst_output =
                "Nothing asked — type what you want to do, then Enter.".to_string();
            return;
        }
        if self.term_asst_busy {
            return;
        }
        self.term_asst_busy = true;
        // Letters stop meaning "type" while the request is in flight; they mean
        // nothing at all until the reply lands (see the `[TERM]ready` handler).
        self.term_asst_typing = false;
        self.term_asst_suggestions.clear();
        self.term_asst_selected = 0;
        self.term_asst_output = format!("Asking {} — {}", self.config.default_model, query);

        // Give the model something real to aim at: the workspace's own layout.
        let root = xencode_context_rs::default_root();
        let mut tops: Vec<String> = self
            .file_tree
            .iter()
            .filter_map(|p| p.split('/').next().map(str::to_string))
            .collect();
        tops.sort();
        tops.dedup();
        tops.truncate(30);
        let prompt = format!(
            "The user wants to do this in a shell, in the workspace at {}:\n\
             {}\n\n\
             Top-level entries in that directory: {}\n\
             git: {}\n\n\
             Reply with a JSON array of at most {} objects and nothing else, each:\n\
             {{\"command\": \"shell command\", \"risk\": \"safe\" or \"destructive\", \"why\": \"one line\"}}",
            root.display(),
            query,
            if tops.is_empty() {
                "(nothing indexed yet)".to_string()
            } else {
                tops.join(", ")
            },
            if self.git_branch.is_empty() {
                "not a git repo".to_string()
            } else {
                format!("branch {}", self.git_branch)
            },
            TERM_SUGGESTION_CAP,
        );
        // Same frozen system head as chat turns, so the instruction that shapes
        // the reply rides in the user message and the cache stays warm.
        let messages = one_shot_messages(&root, prompt);
        let call = SingleShot::from_config(&self.config);

        tokio::spawn(async move {
            match call.ask(&messages).await {
                Err(msg) => {
                    let _ = tx.send(format!("[TERM]error:{msg}"));
                }
                Ok(text) => {
                    let suggestions = parse_term_suggestions(&text);
                    if suggestions.is_empty() {
                        let _ = tx.send(format!(
                            "[TERM]raw:{}",
                            crate::agent_tools::truncate_one_line(&text, 300)
                        ));
                    }
                    for (command, risk, why) in suggestions {
                        let line =
                            serde_json::json!({"command": command, "risk": risk, "why": why});
                        let _ = tx.send(format!("[TERM]suggestion:{}", line));
                    }
                }
            }
            let _ = tx.send("[TERM]ready".to_string());
        });
    }

    /// Run the selected command through the agent's own gate — same policy,
    /// same modal, same hooks and checkpoints. The panel never has a shell of
    /// its own.
    pub fn run_terminal_suggestion(&mut self, tx: mpsc::UnboundedSender<String>) {
        let rows = self.term_visible_rows();
        let Some(&idx) = rows.get(self.term_asst_selected) else {
            self.term_asst_output = "Nothing selected — ask a question first.".to_string();
            return;
        };
        let (command, risk, _) = self.term_asst_suggestions[idx].clone();
        if self.term_asst_busy {
            return;
        }
        self.term_asst_busy = true;
        self.term_asst_output = format!("Waiting for approval: {}", command);
        let ctx = self.approval_ctx();
        let rt = self.task_runtime.clone();
        let root = xencode_context_rs::default_root();
        let call = xencode_providers_rs::ToolCall {
            id: "terminal-panel".to_string(),
            name: "run_command".to_string(),
            arguments: serde_json::json!({ "command": command }),
        };
        tokio::spawn(async move {
            let result =
                crate::agent_tools::execute_tool_call_approved(&rt, &root, &call, &ctx, None).await;
            let line = serde_json::json!({
                "command": command, "risk": risk,
                "result": crate::agent_tools::truncate_one_line(&result, 200),
            });
            let _ = tx.send(format!("[TERM]ran:{}", line));
        });
    }

    /// Start Security Auditor scan simulation.
    /// Scan the workspace with the real pattern scanner over the file list the
    /// context engine's walker produces, so this panel and `xencode analyze`
    /// cannot disagree about what is in the tree.
    pub fn start_security_scan(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.sec_scan_active {
            return;
        }
        self.sec_scan_active = true;
        self.sec_scan_results.clear();
        self.sec_scan_summary = (0, 0, 0, 0);
        self.sec_scan_progress = 0.0;
        self.sec_scan_log.clear();
        let root = xencode_context_rs::default_root();
        self.sec_scan_path = root.display().to_string();
        let _ = tx.send(format!("[SECURITY]note:Scanning {}", self.sec_scan_path));
        tokio::spawn(run_security_scan(root, tx));
    }

    /// Measure what this session actually did. Numbers the app already holds
    /// (turn latency, provider health, llama.cpp timings) are read here; the
    /// process's own CPU/memory and the persisted per-request metrics are
    /// sampled in the background, because CPU needs two reads of `/proc` a
    /// moment apart.
    pub fn start_profiler(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.profiler_running {
            return;
        }
        self.profiler_active = true;
        self.profiler_running = true;
        self.profiler_rows.clear();
        self.profiler_notes.clear();
        self.profiler_gauge_cpu = None;
        self.profiler_gauge_mem = None;
        self.profiler_gauge_mem_total = None;
        self.profiler_gauge_latency = None;

        let uptime = (current_timestamp() - self.session_start_time).max(0.0);
        self.profiler_rows.push((
            "session".to_string(),
            "uptime".to_string(),
            format_uptime(uptime),
        ));
        self.profiler_rows.push((
            "session".to_string(),
            "chat lines".to_string(),
            format!("{}", self.messages.len()),
        ));
        if self.average_latency > 0.0 {
            self.profiler_gauge_latency = Some(self.average_latency);
            self.profiler_rows.push((
                "turn".to_string(),
                "average latency".to_string(),
                format!("{:.0} ms", self.average_latency),
            ));
        } else {
            self.profiler_notes
                .push("no completed turn yet — latency is n/a".to_string());
        }
        if let Some(ts) = &self.last_llamacpp_timings {
            self.profiler_rows.push((
                "llama.cpp".to_string(),
                "generation".to_string(),
                format!("{:.1} tok/s", ts.predicted_per_second),
            ));
            self.profiler_rows.push((
                "llama.cpp".to_string(),
                "prompt eval".to_string(),
                format!("{:.1} tok/s", ts.prompt_per_second),
            ));
            self.profiler_rows.push((
                "llama.cpp".to_string(),
                "tokens in / out".to_string(),
                format!("{} / {}", ts.tokens_evaluated, ts.tokens_generated),
            ));
        }
        let mut health: Vec<(String, String, f64, Option<String>)> = self
            .ollama_health_entries
            .iter()
            .map(|(provider, (status, latency, error))| {
                (provider.clone(), status.clone(), *latency, error.clone())
            })
            .collect();
        health.sort_by(|a, b| a.0.cmp(&b.0));
        for (provider, status, latency, error) in health {
            let value = match error {
                Some(e) if !e.is_empty() => format!("{} — {}", status, e),
                _ => format!("{:.0} ms ({})", latency, status),
            };
            self.profiler_rows
                .push((provider, "last health check".to_string(), value));
        }
        if self.ollama_health_entries.is_empty() {
            self.profiler_notes
                .push("no provider health check this session".to_string());
        }

        let xencode = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
        tokio::spawn(run_profiler(xencode, tx));
    }

    /// The cursor, clamped to a row that exists. An empty list has no row,
    /// which is the honest state before anyone writes `model_profiles`.
    fn selected_profile_index(&self) -> Option<usize> {
        if self.model_profiles.is_empty() {
            None
        } else {
            Some(self.models_selected.min(self.model_profiles.len() - 1))
        }
    }

    /// The profile the cursor is on, if there is one.
    pub fn selected_model_profile(&self) -> Option<&xencode_config_rs::ModelProfile> {
        self.selected_profile_index()
            .map(|idx| &self.model_profiles[idx])
    }

    /// `n`: start a profile from what the session is using right now, so the
    /// panel can create what it can also edit and save.
    pub fn add_model_profile(&mut self) {
        let name = format!("profile {}", self.model_profiles.len() + 1);
        self.model_profiles.push(xencode_config_rs::ModelProfile {
            name: name.clone(),
            model: self.config.default_model.clone(),
            temperature: self.config.llama_cpp_temperature,
            max_tokens: self.config.llama_cpp_max_tokens,
        });
        self.models_selected = self.model_profiles.len() - 1;
        self.models_dirty = true;
        self.models_status = format!("{name} — from this session's settings. Tune it, then s.");
    }

    /// `-`/`+` move the selected profile's temperature. A profile with no
    /// temperature starts from the value the session would send anyway, so the
    /// first step is off a real baseline rather than an invented one.
    pub fn adjust_model_temperature(&mut self, delta: f64) {
        let Some(idx) = self.selected_profile_index() else {
            return;
        };
        let base = self.model_profiles[idx]
            .temperature
            .or(self.config.llama_cpp_temperature)
            .unwrap_or(1.0);
        let next = (base + delta).clamp(0.0, 2.0);
        self.model_profiles[idx].temperature = Some((next * 100.0).round() / 100.0);
        self.models_dirty = true;
        self.models_status.clear();
    }

    /// `←`/`→` step the selected profile's token budget along a fixed ladder —
    /// the numbers a llama.cpp server is actually told to stop at.
    pub fn step_model_max_tokens(&mut self, grow: bool) {
        let Some(idx) = self.selected_profile_index() else {
            return;
        };
        let base = self.model_profiles[idx]
            .max_tokens
            .or(self.config.llama_cpp_max_tokens)
            .unwrap_or(512);
        let last = MODEL_TOKEN_STEPS.len() - 1;
        let at = MODEL_TOKEN_STEPS
            .iter()
            .position(|&step| step >= base)
            .unwrap_or(last);
        let at = if grow {
            (at + 1).min(last)
        } else {
            at.saturating_sub(1)
        };
        self.model_profiles[idx].max_tokens = Some(MODEL_TOKEN_STEPS[at]);
        self.models_dirty = true;
        self.models_status.clear();
    }

    /// `Enter`: this profile's model and sampling become the session's, so the
    /// next turn — chat, agent, or panel — sends them. Deliberately not a disk
    /// write; `s` is the only key that touches config.json.
    pub fn apply_model_profile(&mut self, tx: mpsc::UnboundedSender<String>) {
        let Some(profile) = self.selected_model_profile().cloned() else {
            self.models_status =
                "Nothing to apply: config.json has no model_profiles yet.".to_string();
            return;
        };
        self.config.default_model = profile.model.clone();
        if profile.temperature.is_some() {
            self.config.llama_cpp_temperature = profile.temperature;
        }
        if profile.max_tokens.is_some() {
            self.config.llama_cpp_max_tokens = profile.max_tokens;
        }
        if let Some(pos) = self
            .available_models
            .iter()
            .position(|model| model == &profile.model)
        {
            self.selected_model = pos;
        }
        let mut sent = Vec::new();
        if let Some(temperature) = profile.temperature {
            sent.push(format!("temperature {temperature}"));
        }
        if let Some(max_tokens) = profile.max_tokens {
            sent.push(format!("max tokens {max_tokens}"));
        }
        self.models_status = if sent.is_empty() {
            format!(
                "next turn uses {} with the server's own sampling",
                profile.model
            )
        } else {
            format!("next turn uses {} · {}", profile.model, sent.join(" · "))
        };
        // Same hand-off the model selector does: a llama.cpp server only serves
        // the model it has loaded, so ask it to swap.
        if let Some(inner) = llama_model_target(&profile.model) {
            self.llamacpp_control("switch", Some(inner.to_string()), tx);
        }
    }

    /// `s`: write the panel's profiles into config through the one choke point
    /// every settings write uses. This is the only place a config save reports
    /// why it failed, because here the user asked for the write.
    pub fn save_model_profiles(&mut self) {
        self.config.model_profiles = self.model_profiles.clone();
        if !self.persist_config {
            self.models_status =
                "config persistence is off in this session — nothing was written".to_string();
            return;
        }
        match self.config.save() {
            Ok(()) => {
                self.models_dirty = false;
                self.models_status = format!(
                    "wrote {} profile(s) to config.json",
                    self.model_profiles.len()
                );
            }
            Err(e) => self.models_status = format!("config.json unchanged: {e}"),
        }
    }

    /// `t`: one request carrying exactly this profile's settings, so the row
    /// shows the provider's real answer — or its real error.
    pub fn test_model_profile(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.models_busy {
            return;
        }
        let Some(profile) = self.selected_model_profile().cloned() else {
            self.models_status =
                "Nothing to test: config.json has no model_profiles yet.".to_string();
            return;
        };
        self.models_busy = true;
        self.models_status = format!("asking {}…", profile.model);
        let mut call = SingleShot::from_config(&self.config);
        call.model = profile.model.clone();
        call.llama_opts.temperature = profile.temperature.or(self.config.llama_cpp_temperature);
        call.llama_opts.max_tokens = profile.max_tokens.or(self.config.llama_cpp_max_tokens);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Reply with the single word: ready".to_string().into(),
        }];
        tokio::spawn(async move {
            let token = match call.ask(&messages).await {
                Ok(reply) => format!(
                    "[PROFILE]ok:{}",
                    crate::agent_tools::truncate_one_line(reply.trim(), 160)
                ),
                Err(e) => format!(
                    "[PROFILE]err:{}",
                    crate::agent_tools::truncate_one_line(&e, 200)
                ),
            };
            let _ = tx.send(token);
        });
    }

    /// `Enter` in the Learning panel: queue this workspace's own files as
    /// lessons. What this replaced was one hardcoded Rust ownership lesson —
    /// five sentences, a `calculate_length` snippet, and a quiz whose correct
    /// option was always the first — about code that is not in this repo.
    pub fn start_learning(&mut self, root: std::path::PathBuf, tx: mpsc::UnboundedSender<String>) {
        if self.learn_busy {
            return;
        }
        self.learn_active = true;
        self.learn_root = root;
        match learning_lessons(&self.learn_root) {
            Err(reason) => {
                self.learn_lessons.clear();
                self.learn_current_lesson = 0;
                self.learn_total_lessons = 0;
                self.learn_lesson_title.clear();
                self.learn_code_example.clear();
                self.learn_reset_lesson();
                self.learn_status = reason;
            }
            Ok(lessons) => {
                self.learn_lessons = lessons;
                self.learn_go(1, tx);
            }
        }
    }

    /// Put lesson `n` (1-based, as the panel counts it) on screen and ask the
    /// model about it. Out of range, or a file that cannot be read: the panel
    /// says so and spends no request.
    pub fn learn_go(&mut self, n: usize, tx: mpsc::UnboundedSender<String>) {
        if !self.learn_show(n) {
            return;
        }
        self.learn_ask_current(tx);
    }

    /// The file work, with no provider involved: what the index says this file
    /// declares, and the file's own text. Returns false when there is no lesson
    /// `n` to show, having said why.
    pub fn learn_show(&mut self, n: usize) -> bool {
        let Some((path, declared)) = self.learn_lessons.get(n.wrapping_sub(1)).cloned() else {
            if self.learn_lessons.is_empty() {
                self.learn_status =
                    "Nothing queued — press Enter to build the lessons.".to_string();
            }
            return false;
        };
        self.learn_current_lesson = n;
        self.learn_total_lessons = self.learn_lessons.len();
        self.learn_lesson_title = path.clone();
        self.learn_reset_lesson();

        let source = match std::fs::read_to_string(self.learn_root.join(&path)) {
            Ok(text) => text,
            Err(e) => {
                self.learn_status = format!("The index names {path}, but reading it failed: {e}");
                return false;
            }
        };
        let shown = cap_at_line(&source, LEARN_SOURCE_CAP);
        self.learn_code_example = shown.clone();
        self.learn_content = vec![
            format!(
                "{} declaration(s) the index found in {}.",
                declared.len(),
                path
            ),
            crate::agent_tools::truncate_one_line(&declared.join(" · "), 400),
            if shown.len() == source.len() {
                format!("Whole file sent: {} lines.", source.lines().count())
            } else {
                format!(
                    "First {} of {} bytes sent — the file is longer than the panel shows.",
                    shown.len(),
                    source.len()
                )
            },
        ];
        true
    }

    /// One request per lesson: explain this file, and set a quiz about it with
    /// the answer key the panel grades against. What gets sent is what the
    /// panel is showing — same path, same capped text.
    fn learn_ask_current(&mut self, tx: mpsc::UnboundedSender<String>) {
        let path = self.learn_lesson_title.clone();
        let source = self.learn_code_example.clone();
        if path.is_empty() || source.is_empty() {
            return;
        }
        self.learn_busy = true;
        self.learn_status = format!("asking {} about {path}…", self.config.default_model);
        let prompt = format!(
            "The file `{path}` in this workspace contains:\n\n```rust\n{source}\n```\n\n\
             Teach this file. Reply with ONLY a JSON object shaped like\n\
             {{\"explain\": [\"…\", \"…\"], \"question\": \"…\", \"options\": [\"…\", \"…\", \"…\"], \
             \"answer\": 0, \"why\": \"…\"}}\n\
             where `explain` is 2–4 sentences about code that is actually in the file, `question` asks \
             about this file specifically, `options` is 3–4 choices, `answer` is the 0-based index of \
             the correct choice, and `why` is one sentence on why it is correct.",
        );
        let messages = one_shot_messages(&self.learn_root, prompt);
        let call = SingleShot::from_config(&self.config);
        tokio::spawn(async move {
            let token = match call.ask(&messages).await {
                Ok(reply) => format!("[LEARN]quiz:{reply}"),
                Err(e) => format!("[LEARN]err:{e}"),
            };
            let _ = tx.send(token);
        });
    }

    /// Drop the previous lesson's quiz and explanation; the new file has to
    /// earn both.
    fn learn_reset_lesson(&mut self) {
        self.learn_content.clear();
        self.learn_explain.clear();
        self.learn_quiz_active = false;
        self.learn_quiz_question.clear();
        self.learn_quiz_options.clear();
        self.learn_quiz_selected = 0;
        self.learn_quiz_answered = false;
        self.learn_quiz_correct = false;
        self.learn_quiz_answer = None;
        self.learn_quiz_why.clear();
    }

    /// The model's reply becomes the quiz. A reply with no quiz in it is
    /// reported as that reply rather than filled in with a canned question.
    pub fn learn_apply_quiz(&mut self, reply: &str) {
        self.learn_busy = false;
        match parse_lesson_quiz(reply) {
            Some(lesson) => {
                self.learn_explain = lesson.explain;
                self.learn_quiz_active = true;
                self.learn_quiz_question = lesson.question;
                self.learn_quiz_options = lesson.options;
                self.learn_quiz_answer = Some(lesson.answer);
                self.learn_quiz_why = lesson.why;
                self.learn_status.clear();
            }
            None => {
                self.learn_status = format!(
                    "The model did not answer with a quiz. It said: {}",
                    crate::agent_tools::truncate_one_line(reply.trim(), 200)
                );
            }
        }
    }

    /// Enter on an unanswered quiz: grade against the key the model sent. The
    /// panel has no answer of its own to fall back on.
    pub fn learn_answer_quiz(&mut self) {
        if !self.learn_quiz_active || self.learn_quiz_answered {
            return;
        }
        // A quiz only becomes active together with its key, so one exists here.
        let answer = self.learn_quiz_answer.unwrap_or(usize::MAX);
        self.learn_quiz_answered = true;
        self.learn_quiz_correct = self.learn_quiz_selected == answer;
    }

    /// `p` / `n`: walk the queue the index built.
    pub fn learn_step(&mut self, forward: bool, tx: mpsc::UnboundedSender<String>) {
        if self.learn_busy || self.learn_lessons.is_empty() {
            return;
        }
        let total = self.learn_lessons.len();
        let next = if forward {
            let last = self.learn_current_lesson + 1 > total;
            if last {
                return;
            }
            self.learn_current_lesson + 1
        } else {
            if self.learn_current_lesson <= 1 {
                return;
            }
            self.learn_current_lesson - 1
        };
        self.learn_go(next, tx);
    }

    /// Walk the workspace with the context engine and report what it actually
    /// found per language. What this replaced was five files that do not exist
    /// in this project and a six-row "supported languages" table the scanner
    /// never produced.
    pub fn start_language_scan(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.lang_busy {
            return;
        }
        self.lang_busy = true;
        self.lang_detection_results.clear();
        self.lang_notes.clear();
        let root = xencode_context_rs::default_root();
        self.lang_scan_path = root.display().to_string();
        tokio::spawn(run_language_scan(root, tx));
    }

    /// `Tab` in the panel: the first press starts editing, later ones cycle.
    pub fn cycle_lang_field(&mut self) {
        self.lang_editing = Some(match self.lang_editing {
            None => crate::focus::LangField::Input,
            Some(field) => field.next(),
        });
    }

    pub fn lang_char(&mut self, c: char) {
        let Some(field) = self.lang_editing else {
            return;
        };
        let target = match field {
            crate::focus::LangField::Source => &mut self.lang_translate_source,
            crate::focus::LangField::Target => &mut self.lang_translate_target,
            crate::focus::LangField::Input => &mut self.lang_translate_input,
        };
        target.push(c);
    }

    pub fn lang_backspace(&mut self) {
        let Some(field) = self.lang_editing else {
            return;
        };
        let target = match field {
            crate::focus::LangField::Source => &mut self.lang_translate_source,
            crate::focus::LangField::Target => &mut self.lang_translate_target,
            crate::focus::LangField::Input => &mut self.lang_translate_input,
        };
        target.pop();
    }

    /// One provider call with the text and both languages the panel was told.
    /// The reply is shown as it came back — including an error, which is not
    /// dressed up as a translation.
    pub fn translate_text(&mut self, tx: mpsc::UnboundedSender<String>) {
        let text = self.lang_translate_input.trim().to_string();
        if text.is_empty() {
            self.lang_translate_output =
                "Nothing to translate — Tab selects the Text field, then type.".to_string();
            return;
        }
        if self.lang_busy {
            return;
        }
        self.lang_busy = true;
        self.lang_editing = None;
        self.lang_translate_error = false;
        let named = |value: &str, fallback: &str| {
            let value = value.trim();
            if value.is_empty() {
                fallback.to_string()
            } else {
                value.to_string()
            }
        };
        let prompt = format!(
            "Translate the following text from {} to {}.\nReply with the translation only, no \
             commentary, no quotes, no explanation.\n\n{}",
            named(&self.lang_translate_source, "its own language"),
            named(&self.lang_translate_target, "the same language"),
            text,
        );
        self.lang_translate_output = format!("Asking {}…", self.config.default_model);
        let messages = one_shot_messages(&xencode_context_rs::default_root(), prompt);
        let call = SingleShot::from_config(&self.config);
        tokio::spawn(async move {
            let token = match call.ask(&messages).await {
                Ok(reply) => format!("[TRANS]out:{}", reply.trim_end()),
                Err(e) => format!("[TRANS]error:{e}"),
            };
            let _ = tx.send(token);
        });
    }
    /// Dynamically discover models installed by the user in Ollama and configured cloud models.
    pub fn refresh_models(&mut self, tx: mpsc::UnboundedSender<String>) {
        let ollama_url = self.config.ollama_url.clone();
        let llama_cpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let has_openrouter = self.config.api_keys.openrouter_api_key.is_some();
        let has_gemini = self.config.api_keys.google_gemini_api_key.is_some();
        let has_qwen = self.config.api_keys.qwen_api_key.is_some();
        let current_default = self.config.default_model.clone();

        tokio::spawn(async move {
            let client = OllamaClient::new(&ollama_url, timeout.min(5));
            let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout.min(5));
            let mut models = Vec::new();

            if let Ok(installed) = client.list_models().await {
                for m in installed {
                    if !m.name.contains("embed") {
                        models.push(m.name);
                    }
                }
            }

            // llama.cpp server models
            if let Ok(llama_models) = llama_client.list_models().await {
                for m in llama_models {
                    let prefixed = format!("llamacpp:{}", m.id);
                    if !models.contains(&prefixed) {
                        models.push(prefixed);
                    }
                }
            }

            // Cloud providers if keys are configured
            if has_openrouter {
                models.push("anthropic/claude-3.5-sonnet".to_string());
                models.push("openai/gpt-4o".to_string());
            }
            if has_gemini {
                models.push("google/gemini-1.5-pro".to_string());
                models.push("google/gemini-1.5-flash".to_string());
            }
            if has_qwen {
                models.push("qwen-max".to_string());
                models.push("qwen-plus".to_string());
            }

            // Fallback if no models discovered
            if models.is_empty() && !current_default.is_empty() {
                models.push(current_default);
            }

            if let Ok(serialized) = serde_json::to_string(&models) {
                let _ = tx.send(format!("[MODELS]{}", serialized));
            }
        });
    }

    /// Trigger background health checks across all configured providers.
    pub fn run_health_check(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.health_check_in_progress {
            return;
        }
        self.health_check_in_progress = true;

        let ollama_url = self.config.ollama_url.clone();
        let llama_cpp_url = self.config.llama_cpp_url.clone();
        let timeout = self.config.response_timeout;
        let openrouter_key = self.config.api_keys.openrouter_api_key.clone();
        let qwen_key = self.config.api_keys.qwen_api_key.clone();
        let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
        let default_model = self.config.default_model.clone();

        tokio::spawn(async move {
            // Check Ollama health by listing installed models
            let mut client = OllamaClient::new(&ollama_url, timeout.min(5));
            let start = std::time::Instant::now();

            match client.list_models().await {
                Ok(installed_models) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let chat_models: Vec<String> = installed_models
                        .iter()
                        .filter(|m| !m.name.contains("embed"))
                        .map(|m| m.name.clone())
                        .collect();

                    if !chat_models.is_empty() {
                        // Send refreshed model list to TUI
                        let mut all_models = chat_models.clone();
                        if openrouter_key.is_some() {
                            all_models.push("anthropic/claude-3.5-sonnet".to_string());
                            all_models.push("openai/gpt-4o".to_string());
                        }
                        if gemini_key.is_some() {
                            all_models.push("google/gemini-1.5-pro".to_string());
                            all_models.push("google/gemini-1.5-flash".to_string());
                        }
                        if qwen_key.is_some() {
                            all_models.push("qwen-max".to_string());
                            all_models.push("qwen-plus".to_string());
                        }
                        let _ = tx.send(format!(
                            "[MODELS]{}",
                            serde_json::to_string(&all_models).unwrap_or_default()
                        ));

                        // Test health using actual default_model if local and installed, or first installed model
                        let test_model = if chat_models.contains(&default_model) {
                            &default_model
                        } else {
                            &chat_models[0]
                        };

                        match client.check_health(test_model).await {
                            Ok(health) => {
                                let _ = tx.send(format!(
                                    "[HEALTH]ollama|{}|{}|{}",
                                    health.status,
                                    health.response_time * 1000.0,
                                    health.error_message.unwrap_or_default()
                                ));
                            }
                            Err(e) => {
                                let _ =
                                    tx.send(format!("[HEALTH]ollama|healthy|{}|{}", latency, e));
                            }
                        }
                    } else {
                        let _ = tx.send(format!(
                            "[HEALTH]ollama|healthy|{}|Running (0 models installed)",
                            latency
                        ));
                    }
                }
                Err(e) => {
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let _ = tx.send(format!("[HEALTH]ollama|error|{}|{}", latency, e));
                }
            }

            // Check OpenRouter connectivity (if key is set)
            if let Some(api_key) = openrouter_key {
                let start_or = std::time::Instant::now();
                let openrouter_client = reqwest::Client::new();
                match openrouter_client
                    .get("https://openrouter.ai/api/v1/auth/key")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_or.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() {
                            let _ = tx.send(format!("[HEALTH]openrouter|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!(
                                "[HEALTH]openrouter|error|{}|HTTP {}",
                                latency,
                                resp.status()
                            ));
                        }
                    }
                    Err(e) => {
                        let latency = start_or.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]openrouter|error|{}|{}", latency, e));
                    }
                }
            }

            // Check Qwen DashScope connectivity (if key is set)
            if let Some(api_key) = qwen_key {
                let start_qw = std::time::Instant::now();
                let qwen_client = reqwest::Client::new();
                match qwen_client
                    .get("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_qw.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() || resp.status().as_u16() == 400 {
                            // 400 means the request reached the API but had invalid params (key is valid)
                            let _ = tx.send(format!("[HEALTH]qwen|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!("[HEALTH]qwen|error|{}|HTTP {}", latency, resp.status()));
                        }
                    }
                    Err(e) => {
                        let latency = start_qw.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]qwen|error|{}|{}", latency, e));
                    }
                }
            }

            // Check Gemini connectivity (if key is set)
            if let Some(api_key) = gemini_key {
                let start_ge = std::time::Instant::now();
                let gemini_client = reqwest::Client::new();
                match gemini_client
                    .get(format!(
                        "https://generativelanguage.googleapis.com/v1/models?key={}",
                        api_key
                    ))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        let latency = start_ge.elapsed().as_secs_f64() * 1000.0;
                        if resp.status().is_success() {
                            let _ = tx.send(format!("[HEALTH]gemini|healthy|{}|", latency));
                        } else {
                            let _ = tx.send(format!(
                                "[HEALTH]gemini|error|{}|HTTP {}",
                                latency,
                                resp.status()
                            ));
                        }
                    }
                    Err(e) => {
                        let latency = start_ge.elapsed().as_secs_f64() * 1000.0;
                        let _ = tx.send(format!("[HEALTH]gemini|error|{}|{}", latency, e));
                    }
                }
            }

            // Check llama.cpp connectivity
            let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout.min(5));
            match llama_client.ping().await {
                Ok(resp_time) => {
                    let _ = tx.send(format!("[HEALTH]llamacpp|healthy|{}|", resp_time * 1000.0));
                }
                Err(e) => {
                    let _ = tx.send(format!("[HEALTH]llamacpp|unavailable|0|{}", e));
                }
            }

            let _ = tx.send("[HEALTH_DONE]".to_string());
        });
    }

    /// Auto-start `llama-server` when xencode boots, per config docs
    /// (`llama_cpp_model_path` / `llama_cpp_executable` / `llama_cpp_args`).
    ///
    /// Skips if the server is already answering on `llama_cpp_url`. If no model
    /// path is configured, falls back to discovering a GGUF on disk (see
    /// [`resolve_gguf_model`]), so a `llamacpp:<alias>` default model works out
    /// of the box. The spawned process is session-scoped: it is stopped when
    /// xencode exits (see `Drop for App`), and aborted cleanly if the user
    /// quits before the model finishes loading.
    pub fn maybe_auto_start_llama(&mut self, tx: mpsc::UnboundedSender<String>) {
        // Model alias from the default model id (e.g. `qwen3-4b` from
        // `llamacpp:qwen3-4b`) — used for discovery + `--alias`.
        let alias = self
            .config
            .default_model
            .strip_prefix("llamacpp:")
            .or_else(|| self.config.default_model.strip_prefix("llama.cpp:"))
            .or_else(|| self.config.default_model.strip_prefix("llama:"))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let model_path = self.config.llama_cpp_model_path.clone();
        let exec = self.config.llama_cpp_executable.clone();
        let url = self.config.llama_cpp_url.clone();
        let mut args = self.config.llama_cpp_args.clone();
        if let Some(alias) = &alias {
            if !args.iter().any(|a| a == "--alias") {
                args.push("--alias".to_string());
                args.push(alias.clone());
            }
        }

        let shared = Arc::new(std::sync::Mutex::new(None));
        self.llama_process = Some(shared.clone());
        let cancel = self.llama_cancel.clone();
        let err_tx = tx.clone();
        let ok_tx = tx.clone();

        tokio::spawn(async move {
            // Already running? Attach silently.
            let probe = LlamaCppClient::new(&url, 3);
            if probe.ping().await.is_ok() {
                return;
            }

            let Some(exe) = find_llama_server(if exec.trim().is_empty() {
                None
            } else {
                Some(&exec)
            }) else {
                let _ = err_tx.send(format!(
                    "[LLAMACPP_MSG]⚠️ auto-start skipped: llama-server not on PATH{}",
                    if exec.trim().is_empty() {
                        " (set config llama_cpp_executable)"
                    } else {
                        ""
                    }
                ));
                return;
            };

            // Resolve the GGUF to host (explicit path first, then discovery).
            let explicit = if model_path.trim().is_empty() {
                None
            } else {
                Some(model_path.as_str())
            };
            let resolved = resolve_gguf_model(explicit, alias.as_deref());
            let Some(model_path) = resolved else {
                let _ = err_tx.send(format!(
                    "[LLAMACPP_MSG]⚠️ auto-start skipped: no GGUF model found{}",
                    alias.map(|a| format!(" for `{a}`")).unwrap_or_default()
                ));
                let _ = err_tx.send(
                    "[LLAMACPP_MSG]💡 set config llama_cpp_model_path (xencode config set llama_cpp_model_path <path>) or drop the .gguf into ~/.cache/llama.cpp".to_string(),
                );
                return;
            };

            let port = parse_llama_port(&url);
            let extra: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let mut server = match start_llama_server(&exe, &model_path, port, &extra) {
                Ok(s) => s,
                Err(e) => {
                    let _ = err_tx.send(format!("[LLAMACPP_MSG]⚠️ auto-start failed: {e}"));
                    return;
                }
            };

            // Wait for the server to become healthy (model load can take a while).
            let client = LlamaCppClient::new(&server.base_url, 3);
            let mut ready = false;
            for _ in 0..120 {
                if cancel.load(Ordering::Relaxed) {
                    let _ = server.stop();
                    return; // xencode already exited — don't orphan the server
                }
                if client.ping().await.is_ok() {
                    ready = true;
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            if !ready {
                let _ = server.stop();
                let _ = err_tx.send(
                    "[LLAMACPP_MSG]⚠️ auto-started llama-server did not become ready in time"
                        .to_string(),
                );
                return;
            }
            if cancel.load(Ordering::Relaxed) {
                let _ = server.stop();
                return;
            }

            let pid = server.pid();
            *shared.lock().unwrap() = Some(server);
            let _ = ok_tx.send(format!(
                "[LLAMACPP_MSG]✅ auto-started llama-server on {} (PID {pid}, model {})",
                url, model_path
            ));
            let _ = ok_tx.send("[HEALTH]llamacpp|healthy|0|auto-started".to_string());
            // A new model just came online — refresh the picker so
            // `llamacpp:<name>` shows up without an explicit 'r' press.
            let _ = ok_tx.send("[REFRESH_MODELS]".to_string());
        });
    }

    pub fn submit_review(&mut self, tx: mpsc::UnboundedSender<String>) {
        if self.is_reviewing {
            return;
        }
        if let Some(file_path) = self.file_tree.get(self.selected_file) {
            if let Ok(content) = std::fs::read_to_string(file_path) {
                self.is_reviewing = true;
                self.code_review_output = format!("📝 Reviewing: {}\n\n", file_path);
                self.review_scroll = 0;
                let prompt = format!(
                    "Code review of {}. Identify bugs, security issues, and performance bottlenecks.\n\n```\n{}\n```",
                    file_path, content
                );
                // Same frozen system head as chat turns so reviews follow the
                // project guidelines and reuse the cached prefix.
                let root = xencode_context_rs::default_root();
                let agents = std::fs::read_to_string(root.join("AGENTS.md")).ok();
                let anchor = std::fs::read_to_string(
                    root.join(xencode_context_rs::XENCODE_DIR).join("anchor.md"),
                )
                .ok();
                let messages = vec![
                    ChatMessage {
                        role: "system".to_string(),
                        content: xencode_context_rs::stable_system_text(
                            CTX_SYSTEM,
                            agents.as_deref(),
                            anchor.as_deref(),
                        )
                        .into(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: prompt.into(),
                    },
                ];
                let model = self.config.default_model.clone();
                let ollama_url = self.config.ollama_url.clone();
                let llama_cpp_url = self.config.llama_cpp_url.clone();
                let timeout = self.config.response_timeout;
                let or_key = self.config.api_keys.openrouter_api_key.clone();
                let qwen_key = self.config.api_keys.qwen_api_key.clone();
                let gemini_key = self.config.api_keys.google_gemini_api_key.clone();
                let llama_opts = LlamaCppOptions {
                    temperature: self.config.llama_cpp_temperature,
                    top_k: self.config.llama_cpp_top_k,
                    min_p: self.config.llama_cpp_min_p,
                    max_tokens: self.config.llama_cpp_max_tokens,
                    grammar: None,
                    json_schema: None,
                    mirostat: None,
                };

                tokio::spawn(async move {
                    let client = OllamaClient::new(&ollama_url, timeout);
                    let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout);
                    let manager = ProviderManager::new(client, or_key, qwen_key, gemini_key, None)
                        .with_llama_cpp(llama_client);
                    let _ = manager
                        .generate_stream_with_options(
                            &model,
                            &messages,
                            Some(&llama_opts),
                            |token| {
                                let _ = tx.send(format!("[REVIEW]{}", token));
                            },
                        )
                        .await;
                    if let Some(ts) = manager.last_llamacpp_timings() {
                        if let Ok(json) = serde_json::to_string(&ts) {
                            let _ = tx.send(format!("[TIMINGS]{}", json));
                        }
                    }
                    let _ = tx.send("[REVIEW][DONE]".to_string());
                });
            }
        }
    }
}

/// The agentic turn loop (D1-02), shared by the chat and ByteBot (I2-04):
/// offer the tools on every round but the last, execute requested calls
/// through the permission gate, feed the results back as `AgentTurn` history.
/// The final round is tool-less so a run always ends with a text answer, and
/// every call is gated — a write or a shell command stops at the approval
/// overlay rather than running silently.
/// Create the git worktree a `/spawn` will work in: a sibling of `root`
/// named `<dirname>-spawn-<id>[-<branch>]`, on a fresh branch. Returns the
/// worktree path. Pure enough to test against a temp repo without the TUI.
pub fn spawn_worktree(
    root: &std::path::Path,
    id: u64,
    branch: &str,
) -> Result<std::path::PathBuf, String> {
    let dirname = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace");
    let mut leaf = format!("{dirname}-spawn-{id}");
    if branch != format!("xencode/spawn-{id}") {
        leaf.push('-');
        leaf.push_str(branch);
    }
    let parent = root.parent().unwrap_or(root);
    let path = parent.join(leaf);
    xencode_context_rs::worktree_add(root, &path, Some(branch), true)?;
    Ok(path)
}

/// One assistant step through the provider: the primary model first, then each
/// configured fallback in order (I4-01). A candidate is abandoned only when it
/// fails **before emitting any token** *and* the error is fallback-eligible
/// ([`xencode_providers_rs::retry::is_fallback_eligible`]) — once a token has
/// streamed (or a tool step returned), switching models would duplicate output
/// and double-execute tools, and an error in our own decoder is reproduced by
/// every candidate. Chat sees a `⚠` line for each switch; ByteBot/spawn stay
/// silent and their panels only hear about a total chain failure.
#[allow(clippy::too_many_arguments)]
async fn agent_step_with_fallback(
    manager: &ProviderManager,
    model: &str,
    fallback_models: &[String],
    context_messages: &[ChatMessage],
    history: &[xencode_providers_rs::AgentTurn],
    offer: &[xencode_providers_rs::ToolDefinition],
    llama_opts: &LlamaCppOptions,
    sink: LoopSink,
    tx: &mpsc::UnboundedSender<String>,
    spoken: &mut String,
) -> Result<xencode_providers_rs::AgentStep, xencode_providers_rs::ProviderError> {
    let chain = xencode_providers_rs::retry::fallback_chain(model, fallback_models);
    for (index, candidate) in chain.iter().enumerate() {
        // Fresh per candidate: only a failure with a clean slate is allowed to
        // move on. A token delivered here, even on a failing attempt, fixes the
        // model in place.
        let mut emitted_any = false;
        let attempt = manager
            .generate_stream_with_tools(
                candidate,
                context_messages,
                history,
                offer,
                Some(llama_opts),
                |token| {
                    emitted_any = true;
                    if sink == LoopSink::Chat {
                        let _ = tx.send(token.to_string());
                    } else {
                        spoken.push_str(token);
                    }
                },
            )
            .await;
        match attempt {
            Ok(step) => return Ok(step),
            Err(e) => {
                if !should_advance_fallback(&e, emitted_any) {
                    return Err(e);
                }
                let Some(next) = chain.get(index + 1) else {
                    return Err(e);
                };
                if sink == LoopSink::Chat {
                    let _ = tx.send(format!("[FALLBACK]{candidate} error: {e} · trying {next}"));
                }
            }
        }
    }
    unreachable!("fallback chain is never empty; loop returns inside")
}

/// Whether a failed candidate releases the turn to the next one.
///
/// Two things fix a model in place: output the user has already seen (a
/// switch would duplicate it) and tool work already done (a switch would
/// double-execute it). A [`xencode_providers_rs::ProviderError::Parse`] fixes it
/// too — that is our own decoder failing on bytes we received, so every
/// candidate reproduces it and walking the chain only burns them.
fn should_advance_fallback(err: &xencode_providers_rs::ProviderError, emitted_any: bool) -> bool {
    !emitted_any && xencode_providers_rs::retry::is_fallback_eligible(err)
}

async fn agent_rounds(run: AgentRun, tx: mpsc::UnboundedSender<String>) {
    let AgentRun {
        sink,
        model,
        context_messages,
        approval,
        task_runtime,
        tool_root,
        max_rounds,
        fallback_models,
        ollama_url,
        llama_cpp_url,
        timeout,
        openrouter_key,
        qwen_key,
        gemini_key,
        llama_opts,
    } = run;

    let client = OllamaClient::new(&ollama_url, timeout);
    let llama_client = LlamaCppClient::new(&llama_cpp_url, timeout);
    let manager = ProviderManager::new(client, openrouter_key, qwen_key, gemini_key, None)
        .with_llama_cpp(llama_client)
        .with_request_timeout(timeout);
    let mut tools = xencode_providers_rs::background_tools();
    tools.extend(xencode_providers_rs::advise_tools());
    tools.extend(xencode_providers_rs::file_tools());
    tools.extend(xencode_providers_rs::command_tools());
    tools.extend(xencode_providers_rs::plan_tools());
    // Whatever `/mcp` started, read at the moment the turn begins.
    tools.extend(approval.mcp.definitions());

    let mut history: Vec<xencode_providers_rs::AgentTurn> = Vec::new();
    let mut final_text = String::new();
    let spawn_id = match sink {
        LoopSink::Spawn(id) => Some(id),
        _ => None,
    };
    for round in 0..=max_rounds {
        let offer: &[xencode_providers_rs::ToolDefinition] =
            if round == max_rounds { &[] } else { &tools };
        // Chat streams deltas straight to the transcript; ByteBot wants one
        // row per assistant turn, so its text is collected instead.
        let mut spoken = String::new();
        let step = match agent_step_with_fallback(
            &manager,
            &model,
            &fallback_models,
            &context_messages,
            &history,
            offer,
            &llama_opts,
            sink,
            &tx,
            &mut spoken,
        )
        .await
        {
            Ok(step) => step,
            // Errors leave no partial tool state. The chat finalizes the turn
            // on [DONE] like before; ByteBot has no transcript to bury it in,
            // so the panel says what failed.
            Err(e) => {
                if let Some(id) = spawn_id {
                    let _ = tx.send(format!("{SPAWN_PREFIX}{id}:err:{e}"));
                } else if sink == LoopSink::ByteBot {
                    let _ = tx.send(format!("{BYTEBOT_PREFIX}err:{e}"));
                }
                break;
            }
        };
        if sink == LoopSink::ByteBot || spawn_id.is_some() {
            let text = if step.text.is_empty() {
                std::mem::take(&mut spoken)
            } else {
                step.text.clone()
            };
            if let Some(id) = spawn_id {
                // The last round's answer becomes the finish report; a run that
                // broke out on an error leaves it empty and says so via err:.
                final_text = text.clone();
                if !text.trim().is_empty() {
                    let _ = tx.send(format!(
                        "{SPAWN_PREFIX}{id}:log:{}",
                        crate::agent_tools::truncate_one_line(&text, 200)
                    ));
                }
            } else if !text.trim().is_empty() {
                let _ = tx.send(format!(
                    "{BYTEBOT_PREFIX}log:{}",
                    crate::agent_tools::truncate_one_line(&text, 200)
                ));
            }
        }
        if step.tool_calls.is_empty() {
            break;
        }
        history.push(xencode_providers_rs::AgentTurn::Assistant {
            text: step.text.clone(),
            calls: step.tool_calls.clone(),
        });
        for call in &step.tool_calls {
            let summary = crate::agent_tools::summarize_call(call);
            match sink {
                LoopSink::Chat => {
                    let _ = tx.send(format!("[TOOL]→ {summary}"));
                }
                LoopSink::ByteBot => {
                    let _ = tx.send(format!("{BYTEBOT_PREFIX}call:{summary}"));
                }
                LoopSink::Spawn(id) => {
                    let _ = tx.send(format!("{SPAWN_PREFIX}{id}:call:{summary}"));
                }
            }
            let result = crate::agent_tools::execute_tool_call_approved(
                &task_runtime,
                &tool_root,
                call,
                &approval,
                Some(&approval.mcp),
            )
            .await;
            let outcome = crate::agent_tools::call_outcome(&result);
            if sink == LoopSink::Chat && outcome == crate::agent_tools::CallOutcome::Refused {
                // A policy refusal never reaches the overlay, so it needs its
                // own transcript line or it would be invisible outside the
                // model's context.
                let _ = tx.send(format!(
                    "[TOOL]✗ {} · refused: outside the workspace",
                    crate::agent_tools::approval_summary(call)
                ));
            }
            match sink {
                LoopSink::Chat => {
                    let _ = tx.send(format!(
                        "[TOOL]← {}",
                        crate::agent_tools::truncate_one_line(&result, 120)
                    ));
                }
                LoopSink::ByteBot => {
                    let _ = tx.send(format!("{BYTEBOT_PREFIX}done:{}", outcome.label()));
                }
                LoopSink::Spawn(id) => {
                    let _ = tx.send(format!("{SPAWN_PREFIX}{id}:done:{}", outcome.label()));
                }
            }
            history.push(xencode_providers_rs::AgentTurn::ToolResult {
                id: call.id.clone(),
                content: result,
            });
        }
    }
    // Report llama.cpp tok/s stats if this was a llama.cpp request
    if let Some(ts) = manager.last_llamacpp_timings() {
        if let Ok(json) = serde_json::to_string(&ts) {
            let _ = tx.send(format!("[TIMINGS]{}", json));
        }
    }
    let _ = tx.send(match sink {
        LoopSink::Chat => "[DONE]".to_string(),
        LoopSink::ByteBot => "[BYTEBOT_DONE]".to_string(),
        LoopSink::Spawn(id) => format!("{SPAWN_PREFIX}{id}:finish:{final_text}"),
    });
}

impl<'a> Default for App<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract the inner model id from a llama.cpp-prefixed model selector entry.
pub(crate) fn llama_model_target(model: &str) -> Option<&str> {
    for prefix in ["llamacpp:", "llama.cpp:", "llama:"] {
        if let Some(rest) = model.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

/// Stops the session-scoped llama-server when xencode exits. Runs on every
/// teardown path (quitting via `q`, `Ctrl+C`, or an error return), and aborts a
/// still-loading auto-start so it can't orphan a process behind the app.
impl<'a> Drop for App<'a> {
    fn drop(&mut self) {
        self.llama_cancel.store(true, Ordering::Relaxed);
        if let Some(shared) = self.llama_process.take() {
            if let Some(mut server) = shared.lock().unwrap().take() {
                let _ = server.stop();
            }
        }
    }
}

/// Extract the port from a llama.cpp base URL, e.g. `http://127.0.0.1:8080` → `8080`.
pub fn parse_llama_port(url: &str) -> u16 {
    let without_scheme = url
        .trim()
        .strip_prefix("http://")
        .or_else(|| url.trim().strip_prefix("https://"))
        .unwrap_or(url.trim());
    let host_and_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    host_and_port
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .filter(|p| *p > 0)
        .unwrap_or(8080)
}

pub async fn run_app<B: Backend>(terminal: &mut Terminal<B>) -> io::Result<()> {
    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Query installed Ollama models and check provider health immediately on startup
    app.refresh_models(tx.clone());
    app.run_health_check(tx.clone());
    // If llama.cpp is configured but not running, spawn it (attaches if it is).
    app.maybe_auto_start_llama(tx.clone());

    // Real-time file watcher: every non-ignored change is reported as a
    // `[WATCH]<kind>|<path>` token. The drain loop only surfaces warnings for
    // files the session actually cares about (tracked/attached/open), so a
    // large workspace does not spam the chat.
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            let root = xencode_context_rs::default_root();
            let Ok(mut watcher) = xencode_context_rs::WorkspaceWatcher::spawn(&root, &[]) else {
                return;
            };
            loop {
                let batch = watcher.next_batch(Duration::from_millis(250));
                if batch.is_empty() {
                    continue;
                }
                for ev in batch {
                    let kind = match ev.kind {
                        xencode_context_rs::WatchKind::Created => "created",
                        xencode_context_rs::WatchKind::Modified => "modified",
                        xencode_context_rs::WatchKind::Removed => "removed",
                    };
                    if tx.send(format!("[WATCH]{}|{}", kind, ev.path)).is_err() {
                        return;
                    }
                }
            }
        });
    }

    loop {
        crate::toast::prune(&mut app.toasts, current_timestamp());
        terminal.draw(|f| ui::draw(f, &mut app))?;

        // Drain async messages
        while let Ok(token) = rx.try_recv() {
            if let Some(body) = token.strip_prefix("[REVIEW]") {
                app.append_review(body);
            } else if let Some(body) = token.strip_prefix("[SPAWN]") {
                // `<id>:<event>` — a `/spawn` subagent reporting in (I3-03).
                if let Some((id, rest)) = body.split_once(':') {
                    if let Ok(id) = id.parse::<u64>() {
                        app.spawn_event(id, rest);
                    }
                }
            } else if let Some(body) = token.strip_prefix("[BYTEBOT]") {
                app.bytebot_event(body);
            } else if token == "[BYTEBOT_DONE]" {
                app.bytebot_running = false;
                // Whatever came back is what there is: the bar and the closing
                // line read the step rows, so an aborted run cannot claim 100%.
                app.bytebot_progress = bytebot_progress(&app.bytebot_steps);
                let done = app
                    .bytebot_steps
                    .iter()
                    .filter(|(_, status)| status == "done")
                    .count();
                app.bytebot_log.push(format!(
                    "■ run over: {done}/{} call(s) completed",
                    app.bytebot_steps.len()
                ));
            } else if token == "[INIT_DONE]" {
                app.init_running = false;
                app.init_progress = 1.0;
            } else if let Some(body) = token.strip_prefix("[INIT]") {
                if body.starts_with("step:") {
                    let parts: Vec<&str> = body.splitn(4, ':').collect();
                    if parts.len() >= 4 {
                        let idx = parts[1].parse::<usize>().unwrap_or(0);
                        if idx < app.init_steps.len() {
                            app.init_steps[idx].1 = parts[2].to_string();
                        }
                    }
                } else if body.starts_with("progress:") {
                    if let Some(pct) = body.strip_prefix("progress:") {
                        app.init_progress = pct.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("log:") {
                    if let Some(msg) = body.strip_prefix("log:") {
                        app.init_log.push(msg.to_string());
                    }
                }
            } else if token == "[CTX_START]" || token == "[ADVISE_START]" {
                // Both open a fresh assistant message that the matching
                // [CTX]/[ADVISE] tokens populate line by line.
                app.messages.push(UiMessage {
                    role: "assistant".to_string(),
                    content: String::new(),
                });
            } else if let Some(body) = token.strip_prefix("[ADVISE]") {
                if let Some(last) = app.messages.last_mut() {
                    if last.role == "assistant" {
                        if !last.content.is_empty() {
                            last.content.push('\n');
                        }
                        last.content.push_str(body);
                    }
                }
            } else if let Some(body) = token.strip_prefix("[CTX]") {
                if let Some(last) = app.messages.last_mut() {
                    if last.role == "assistant" {
                        if !last.content.is_empty() {
                            last.content.push('\n');
                        }
                        last.content.push_str(body);
                    }
                }
            } else if let Some(body) = token.strip_prefix("[COLLAB]") {
                app.apply_collab_token(body);
            } else if let Some(body) = token.strip_prefix("[HEALTH]") {
                let parts: Vec<&str> = body.splitn(4, '|').collect();
                if parts.len() >= 3 {
                    let provider = parts[0].to_string();
                    let status = parts[1].to_string();
                    let latency = parts[2].parse::<f64>().unwrap_or(0.0);
                    let error = if parts.len() > 3 && !parts[3].is_empty() {
                        Some(parts[3].to_string())
                    } else {
                        None
                    };
                    app.ollama_health_entries
                        .insert(provider, (status.clone(), latency, error));
                    // Update average latency across all providers
                    if status == "healthy" {
                        let total: f64 = app
                            .ollama_health_entries
                            .values()
                            .map(|(s, l, _)| if s == "healthy" { *l } else { 0.0 })
                            .sum();
                        let count = app
                            .ollama_health_entries
                            .values()
                            .filter(|(s, _, _)| s == "healthy")
                            .count() as f64;
                        app.average_latency = if count > 0.0 { total / count } else { 0.0 };
                    }
                }
            } else if token == "[REFRESH_MODELS]" {
                app.refresh_models(tx.clone());
            } else if let Some(body) = token.strip_prefix("[LLAMACPP_MSG]") {
                app.llamacpp_action_msg = body.to_string();
            } else if let Some(body) = token.strip_prefix("[VOICE]") {
                if let Some(s) = body.strip_prefix("status:") {
                    let new_status = s.to_string();
                    if new_status == "idle" {
                        app.voice_finish();
                    } else {
                        app.voice_status = new_status;
                    }
                } else if let Some(l) = body.strip_prefix("level:") {
                    app.voice_apply_level(l);
                } else if let Some(p) = body.strip_prefix("peak:") {
                    app.voice_peak = p.trim().parse::<f64>().unwrap_or(app.voice_peak);
                } else if let Some(c) = body.strip_prefix("clip:") {
                    app.voice_apply_clip(c);
                } else if let Some(t) = body.strip_prefix("transcript:") {
                    app.voice_apply_transcript(t);
                } else if let Some(n) = body.strip_prefix("note:") {
                    app.voice_apply_note(n);
                } else if let Some(e) = body.strip_prefix("err:") {
                    app.voice_apply_error(e);
                }
            } else if let Some(body) = token.strip_prefix("[TERM]") {
                if let Some(json) = body.strip_prefix("suggestion:") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
                        app.term_asst_typing = false;
                        app.term_asst_suggestions.push((
                            v["command"].as_str().unwrap_or_default().to_string(),
                            v["risk"].as_str().unwrap_or_default().to_string(),
                            v["why"].as_str().unwrap_or_default().to_string(),
                        ));
                    }
                } else if let Some(json) = body.strip_prefix("ran:") {
                    app.term_asst_busy = false;
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
                        let command = v["command"].as_str().unwrap_or_default().to_string();
                        let risk = v["risk"].as_str().unwrap_or_default().to_string();
                        let result = v["result"].as_str().unwrap_or_default().to_string();
                        app.term_asst_history
                            .push((command.clone(), risk, result.clone()));
                        app.term_asst_output = format!("{} — {}", command, result);
                    }
                } else if let Some(msg) = body.strip_prefix("raw:") {
                    app.term_asst_typing = true;
                    app.term_asst_output =
                        format!("The model did not answer with commands. It said: {}", msg);
                } else if let Some(msg) = body.strip_prefix("error:") {
                    app.term_asst_typing = true;
                    // One line, because the status box is one line tall.
                    app.term_asst_output = crate::agent_tools::truncate_one_line(msg, 300);
                } else if body == "ready" {
                    app.term_asst_busy = false;
                    if app.term_asst_suggestions.is_empty() {
                        app.term_asst_typing = true;
                    }
                }
            } else if let Some(body) = token.strip_prefix("[LANG]") {
                if let Some(json) = body.strip_prefix("row:") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(json) {
                        app.lang_detection_results.push((
                            v["language"].as_str().unwrap_or_default().to_string(),
                            v["files"].as_u64().unwrap_or(0),
                            v["lines"].as_u64().unwrap_or(0),
                            v["share"].as_f64().unwrap_or(0.0),
                        ));
                    }
                } else if let Some(note) = body.strip_prefix("note:") {
                    app.lang_notes.push(note.to_string());
                } else if let Some(why) = body.strip_prefix("failed:") {
                    app.lang_notes
                        .push(format!("the walk did not finish: {why}"));
                    app.lang_busy = false;
                } else if body == "done" {
                    app.lang_busy = false;
                }
            } else if let Some(body) = token.strip_prefix("[TRANS]") {
                app.lang_busy = false;
                // Both arms write the same field on purpose: what came back,
                // answer or error, is the panel's output line.
                if let Some(reply) = body.strip_prefix("out:") {
                    app.lang_translate_error = false;
                    app.lang_translate_output = reply.to_string();
                } else if let Some(err) = body.strip_prefix("error:") {
                    app.lang_translate_error = true;
                    app.lang_translate_output = err.to_string();
                }
            } else if let Some(body) = token.strip_prefix("[LEARN]") {
                if let Some(reply) = body.strip_prefix("quiz:") {
                    app.learn_apply_quiz(reply);
                } else if let Some(err) = body.strip_prefix("err:") {
                    app.learn_busy = false;
                    app.learn_status = format!(
                        "provider said: {}",
                        crate::agent_tools::truncate_one_line(err, 200)
                    );
                }
            } else if let Some(body) = token.strip_prefix("[PROFILE]") {
                app.models_busy = false;
                // The reply and the failure share one line on purpose: the
                // panel's status is the provider's own words either way.
                if let Some(reply) = body.strip_prefix("ok:") {
                    app.models_status = format!("reply: {reply}");
                } else if let Some(err) = body.strip_prefix("err:") {
                    app.models_status = format!("test failed: {err}");
                }
            } else if let Some(body) = token.strip_prefix("[SECURITY]") {
                if body.starts_with("progress:") {
                    if let Some(p) = body.strip_prefix("progress:") {
                        app.sec_scan_progress = p.trim().parse::<f64>().unwrap_or(0.0);
                    }
                } else if body.starts_with("finding:") {
                    if let Some(f) = body.strip_prefix("finding:") {
                        let parts: Vec<&str> = f.splitn(4, '|').collect();
                        if parts.len() >= 4 {
                            let severity = parts[0].to_string();
                            let category = parts[1].to_string();
                            let location = parts[2].to_string();
                            let detail = parts[3].to_string();
                            app.sec_scan_results.push((
                                severity.clone(),
                                category.clone(),
                                location.clone(),
                            ));
                            app.sec_scan_log.push(detail);
                            // Update summary counts
                            let (mut c, mut h, mut m, mut l) = app.sec_scan_summary;
                            match severity.as_str() {
                                "Critical" => c += 1,
                                "High" => h += 1,
                                "Medium" => m += 1,
                                _ => l += 1,
                            }
                            app.sec_scan_summary = (c, h, m, l);
                        }
                    }
                } else if let Some(msg) = body.strip_prefix("note:") {
                    app.sec_scan_log.push(msg.to_string());
                } else if let Some(msg) = body.strip_prefix("failed:") {
                    app.sec_scan_log.push(format!("scan failed: {}", msg));
                    app.sec_scan_active = false;
                } else if let Some(msg) = body.strip_prefix("done:") {
                    app.sec_scan_progress = 1.0;
                    // The scan's own totals win over the per-line count: the
                    // list is capped on screen, the findings are not.
                    let (summary, text) = match msg.split_once('|') {
                        Some((text, counts)) => {
                            let c: Vec<u32> = counts
                                .split(',')
                                .filter_map(|v| v.trim().parse().ok())
                                .collect();
                            let totals = if c.len() == 4 {
                                (c[0], c[1], c[2], c[3])
                            } else {
                                app.sec_scan_summary
                            };
                            (totals, text)
                        }
                        None => (app.sec_scan_summary, msg),
                    };
                    app.sec_scan_summary = summary;
                    app.sec_scan_log.push(text.to_string());
                    app.sec_scan_active = false;
                }
            } else if let Some(body) = token.strip_prefix("[PROFILER]") {
                if body.starts_with("gauge:") {
                    if let Some(g) = body.strip_prefix("gauge:") {
                        let parts: Vec<&str> = g.splitn(2, '|').collect();
                        if parts.len() >= 2 {
                            let val = parts[1].parse::<f64>().ok();
                            match parts[0] {
                                "cpu" => app.profiler_gauge_cpu = val,
                                "mem" => app.profiler_gauge_mem = val,
                                "memtotal" => app.profiler_gauge_mem_total = val,
                                "latency" => app.profiler_gauge_latency = val,
                                _ => {}
                            }
                        }
                    }
                } else if let Some(r) = body.strip_prefix("row:") {
                    let parts: Vec<&str> = r.splitn(3, '|').collect();
                    if parts.len() == 3 {
                        app.profiler_rows.push((
                            parts[0].to_string(),
                            parts[1].to_string(),
                            parts[2].to_string(),
                        ));
                    }
                } else if let Some(msg) = body.strip_prefix("note:") {
                    app.profiler_notes.push(msg.to_string());
                } else if let Some(msg) = body.strip_prefix("failed:") {
                    app.profiler_notes
                        .push(format!("profiling failed: {}", msg));
                    app.profiler_running = false;
                } else if body == "done" {
                    app.profiler_running = false;
                }
            } else if let Some(body) = token.strip_prefix("[MODELS]") {
                if let Ok(models) = serde_json::from_str::<Vec<String>>(body) {
                    if !models.is_empty() {
                        app.available_models = models;
                        if let Some(pos) = app
                            .available_models
                            .iter()
                            .position(|m| m == &app.config.default_model)
                        {
                            app.selected_model = pos;
                        } else {
                            // If current default_model is not installed, select first installed model from Ollama
                            if let Some(first) = app.available_models.first().cloned() {
                                app.config.default_model = first;
                                app.selected_model = 0;
                                app.save_config();
                            }
                        }
                    }
                }
            } else if let Some(body) = token.strip_prefix("[CTXSTATS]") {
                let parts: Vec<&str> = body.splitn(2, '|').collect();
                if parts.len() == 2 {
                    app.last_ctx_total_tokens = parts[0].parse().unwrap_or(0);
                    app.last_ctx_retrieved_files = parts[1].parse().unwrap_or(0);
                }
            } else if let Some(body) = token.strip_prefix("[LLAMACPP]") {
                app.llamacpp_action_msg = body.to_string();
            } else if let Some(body) = token.strip_prefix("[TIMINGS]") {
                if let Ok(ts) = serde_json::from_str::<LlamaCppTimings>(body) {
                    app.last_llamacpp_timings = Some(ts.clone());
                    // Record a §13 metrics row: cached = prompt_total − actually
                    // evaluated. prompt_total comes from the last /ctx assembly,
                    // evaluated from llama.cpp — a ~0 cached_tokens with a large
                    // stable prefix means prefix stability broke somewhere.
                    let root = xencode_context_rs::default_root();
                    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
                    let profile = xencode_context_rs::HardwareProfile::Balanced;
                    let m = xencode_context_rs::RequestMetrics::from_timings(
                        profile.name(),
                        profile.ctx_tokens() as u32,
                        app.last_ctx_total_tokens.min(u32::MAX as u64) as u32,
                        ts.tokens_evaluated.min(u32::MAX as u64) as u32,
                        ts.tokens_generated.min(u32::MAX as u64) as u32,
                        ts.predicted_per_second as f32,
                        ts.prompt_per_second as f32,
                        app.last_ctx_retrieved_files,
                    );
                    let _ = xencode_context_rs::append_metrics(&xencode, &m);
                }
            } else if token == "[HEALTH_DONE]" {
                app.health_check_in_progress = false;
                app.last_health_check = current_timestamp();
            } else if let Some(body) = token.strip_prefix("[GIT_COMMIT_OK]") {
                app.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!("✓ Commit: {body}"),
                });
                app.refresh_git();
            } else if let Some(body) = token.strip_prefix("[GIT_COMMIT_ERR]") {
                app.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!("✗ Commit failed: {body}"),
                });
                app.refresh_git();
            } else if let Some(body) = token.strip_prefix("[TASKS]") {
                app.handle_tasks_command(body);
            } else if let Some(body) = token.strip_prefix("[TOOL]") {
                // Tool-loop lines arrive mid-stream (D1-02); the next token
                // opens a fresh assistant bubble, so each round stays visible.
                app.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!("⚙{body}"),
                });
            } else if let Some(body) = token.strip_prefix("[MCP]") {
                // `/mcp` reports (I3-01) arrive from the connect/status task.
                app.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!("◈ {body}"),
                });
            } else if let Some(body) = token.strip_prefix("[FALLBACK]") {
                // Provider fallback chain (I4-01): the primary model failed
                // before emitting anything, so the turn is retried on the next
                // configured model. `⚠` keeps it a system note, not a bubble.
                app.messages.push(UiMessage {
                    role: "system".to_string(),
                    content: format!("⚠ {body}"),
                });
            } else if let Some(body) = token.strip_prefix("[WATCH]") {
                app.handle_watch_event(body);
            } else {
                app.append_generation(&token);
            }
        }

        // Approval prompts from the tool loop (I1-03): queue them; the
        // modal overlay answers the front one and wakes its task.
        if let Some(arx) = app.approval_rx.as_mut() {
            while let Ok((request, responder)) = arx.try_recv() {
                app.approval_queue.push_back((request, responder));
            }
        }

        // Poll events (~30fps)
        if event::poll(Duration::from_millis(33))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Dispatch lives in keymap.rs (E6-01): modal help overlay,
                    // global Ctrl chords, then per-focus handlers.
                    if crate::keymap::handle_key(&mut app, key, &tx) == crate::keymap::KeyFlow::Quit
                    {
                        return Ok(());
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => match app.focus {
                        FocusArea::ChatInput => {
                            app.chat_scroll = app.chat_scroll.saturating_add(3);
                        }
                        FocusArea::CodeEditor => {
                            app.editor.scroll((-3, 0));
                        }
                        FocusArea::FileExplorer => {
                            if app.selected_file >= 3 {
                                app.selected_file -= 3;
                            } else {
                                app.selected_file = 0;
                            }
                        }
                        FocusArea::FeatureNavigator => {
                            if app.feature_nav_selected >= 3 {
                                app.feature_nav_selected -= 3;
                            } else {
                                app.feature_nav_selected = 0;
                            }
                        }
                        FocusArea::ProviderHealth => {
                            if app.provider_health_scroll >= 3 {
                                app.provider_health_scroll -= 3;
                            } else {
                                app.provider_health_scroll = 0;
                            }
                        }
                        FocusArea::SecurityAuditor => {
                            if app.security_scroll >= 3 {
                                app.security_scroll -= 3;
                            } else {
                                app.security_scroll = 0;
                            }
                        }
                        FocusArea::ReviewDashboard => {
                            app.review_dash.scroll_by(-3);
                        }
                        FocusArea::TaskManager => {
                            if app.tasks_detail {
                                app.tasks_scroll = app.tasks_scroll.saturating_sub(3);
                            } else {
                                app.tasks_selected = app.tasks_selected.saturating_sub(3);
                            }
                        }
                        // E2-05: wheel drives the same state as ↑ for panels
                        // that have a cursor/scroll offset but lacked wheel.
                        // Remaining panels are single-screen with nothing to scroll.
                        FocusArea::Settings => {
                            if app.settings_cursor > 0 {
                                app.settings_cursor -= 1;
                            }
                        }
                        FocusArea::ModelSelector => {
                            if app.selected_model > 0 {
                                app.selected_model -= 1;
                            }
                        }
                        FocusArea::CustomModels => {
                            if app.models_selected > 0 {
                                app.models_selected -= 1;
                            }
                        }
                        FocusArea::CodeReview => {
                            app.review_scroll = app.review_scroll.saturating_sub(1);
                        }
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.focus {
                        FocusArea::ChatInput => {
                            app.chat_scroll = app.chat_scroll.saturating_sub(3);
                        }
                        FocusArea::CodeEditor => {
                            app.editor.scroll((3, 0));
                        }
                        FocusArea::FileExplorer => {
                            app.selected_file =
                                (app.selected_file + 3).min(app.file_tree.len().saturating_sub(1));
                        }
                        FocusArea::FeatureNavigator => {
                            app.feature_nav_selected = (app.feature_nav_selected + 3)
                                .min(FEATURE_LIST.len().saturating_sub(1));
                        }
                        FocusArea::ProviderHealth => {
                            app.provider_health_scroll += 3;
                        }
                        FocusArea::SecurityAuditor => {
                            app.security_scroll += 3;
                        }
                        FocusArea::ReviewDashboard => {
                            app.review_dash.scroll_by(3);
                        }
                        FocusArea::TaskManager => {
                            if app.tasks_detail {
                                app.tasks_scroll += 3;
                            } else if let Some(tasks) = app.tasks_snapshot() {
                                app.tasks_selected =
                                    (app.tasks_selected + 3).min(tasks.len().saturating_sub(1));
                            }
                        }
                        FocusArea::Settings => {
                            if app.settings_cursor + 1 < crate::focus::SETTINGS_ITEMS.len() {
                                app.settings_cursor += 1;
                            }
                        }
                        FocusArea::ModelSelector => {
                            if app.selected_model + 1 < app.available_models.len() {
                                app.selected_model += 1;
                            }
                        }
                        FocusArea::CustomModels => {
                            if app.models_selected + 1 < app.model_profiles.len() {
                                app.models_selected += 1;
                            }
                        }
                        FocusArea::CodeReview => {
                            app.review_scroll += 1;
                        }
                        _ => {}
                    },
                    MouseEventKind::Down(MouseButton::Left) => {
                        let size = terminal.size()?;
                        // Same outer split as ui::draw: header(1) + body + status(1).
                        let body_area = ratatui::layout::Rect::new(
                            0,
                            1,
                            size.width,
                            size.height.saturating_sub(2),
                        );
                        let body_layout = crate::layout::compute_layout(
                            body_area,
                            &app.config.layout,
                            app.show_terminal,
                            app.last_body_focus,
                        );
                        match body_layout.hit_test(mouse.column) {
                            Some(FocusArea::FileExplorer) => {
                                app.focus = FocusArea::FileExplorer;
                                let row = mouse.row.saturating_sub(2) as usize;
                                if row < app.file_tree.len() {
                                    app.selected_file = row;
                                }
                            }
                            Some(target) => app.focus = target,
                            None => {}
                        }
                    }
                    _ => {}
                },
                Event::Resize(width, height) => {
                    ui::clamp_scrolls_on_resize(&mut app, width, height);
                }
                _ => {}
            }
        } else if app.is_generating
            || app.is_reviewing
            || app.health_check_in_progress
            || app.bytebot_running
            || app.voice_busy
            || app.collab_sync_status == "connecting"
            || app.sec_scan_active
            || app.profiler_running
        {
            app.spinner_tick = app.spinner_tick.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        cap_at_line, first_output_line, format_advise_report, format_watch_warning,
        learning_lessons, live_refresh_snapshot, parse_lesson_quiz, parse_llama_port,
        parse_porcelain_z, parse_term_suggestions, parse_voice_level, watch_warning_for, App,
        FocusArea, LoopSink, SpawnRecord,
    };
    use std::collections::HashSet;
    use tokio::sync::mpsc;
    use xencode_context_rs::init_project;
    use xencode_core_rs::{scan_workspace, ScanOptions, TaskStatus};

    /// I2-01: `/rewind` is the user's undo for what the agent wrote. The
    /// checkpoint is recorded by the real gated path, not seeded by hand.
    #[tokio::test]
    async fn rewind_command_undoes_agent_writes_and_reports_them() {
        let dir = std::env::temp_dir().join(format!("xencode-rewind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("keep.txt"), "mine\n").unwrap();

        let mut app = App::for_tests();
        let (prompts, _rx) = mpsc::unbounded_channel();
        let ctx = crate::agent_tools::ApprovalCtx {
            mode: crate::agent_tools::ApprovalMode::AllAllow,
            grants: app.agent_grants.clone(),
            prompts,
            checkpoints: app.checkpoints.clone(),
            turn: app.checkpoints.begin_turn(),
            command_timeout: crate::agent_tools::DEFAULT_COMMAND_TIMEOUT,
            plan: app.agent_plan.clone(),
            mcp: app.mcp.clone(),
            hooks: app.config.agent_hooks.clone(),
        };
        let call = xencode_providers_rs::ToolCall {
            id: "c1".to_string(),
            name: "edit_file".to_string(),
            arguments: serde_json::json!({"path": "keep.txt", "old": "mine", "new": "theirs"}),
        };
        let wrote = crate::agent_tools::execute_tool_call_approved(
            &app.task_runtime,
            &dir,
            &call,
            &ctx,
            Some(&app.mcp),
        )
        .await;
        assert!(wrote.starts_with("edited keep.txt"), "{wrote}");
        assert!(dir.join("keep.txt").exists());

        app.handle_rewind_command("/rewind");
        assert_eq!(
            std::fs::read_to_string(dir.join("keep.txt")).unwrap(),
            "mine\n"
        );
        let line = app.messages.last().unwrap();
        assert_eq!(line.role, "system");
        assert!(
            line.content.contains("Rewound 1 agent turn") && line.content.contains("keep.txt"),
            "{:?}",
            line.content
        );

        // Honest about having nothing left, rather than pretending to undo.
        app.handle_rewind_command("/rewind");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("Nothing to rewind"));

        // A bad argument is usage, not a silent no-op.
        app.handle_rewind_command("/rewind lots");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("usage: /rewind"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn rewind_refuses_to_fight_a_running_generation() {
        let mut app = App::for_tests();
        app.is_generating = true;
        let before = app.messages.len();
        app.handle_rewind_command("/rewind");
        assert_eq!(
            app.messages.len(),
            before,
            "mid-generation rewinds must not touch files"
        );
        assert!(
            app.toasts
                .iter()
                .any(|toast| toast.message.contains("while the agent is working")),
            "the refusal has to be visible"
        );

        // I2-04: ByteBot writes through the same gate, so it counts too.
        let mut app = App::for_tests();
        app.bytebot_running = true;
        app.handle_rewind_command("/rewind");
        assert!(app
            .toasts
            .iter()
            .any(|toast| toast.message.contains("while the agent is working")));
    }

    /// I3-01: `/mcp` is the only way servers start, so with none configured it
    /// must say so — and status/stop report the honest empty state, in the
    /// model's stream rather than a dead-end.
    #[tokio::test]
    async fn mcp_command_without_servers_is_honest_about_it() {
        let mut app = App::for_tests();
        assert!(
            app.config.mcp_servers.is_empty(),
            "the default config declares no servers"
        );

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        app.handle_mcp_command("/mcp", tx.clone());
        assert!(
            app.messages
                .last()
                .unwrap()
                .content
                .contains("No MCP servers configured"),
            "{}",
            app.messages.last().unwrap().content
        );

        app.handle_mcp_command("/mcp status", tx.clone());
        let said = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("status answer must arrive")
            .expect("status channel stays open");
        assert!(said.starts_with("[MCP]"), "{said}");
        assert!(said.contains("no MCP servers running"), "{said}");

        app.handle_mcp_command("/mcp stop", tx.clone());
        let said = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("stop answer must arrive")
            .expect("stop channel stays open");
        assert!(said.starts_with("[MCP]"), "{said}");
        assert!(said.contains("no MCP servers running"), "{said}");

        app.handle_mcp_command("/mcp nope", tx);
        assert!(
            app.messages.last().unwrap().content.contains("usage: /mcp"),
            "{}",
            app.messages.last().unwrap().content
        );
    }

    /// I3-01: the slash dispatcher hands `/mcp` off before any provider work,
    /// so the command sits in the transcript like a user turn and the servers'
    /// answer comes back on the same channel a generation would use.
    #[tokio::test]
    async fn mcp_slash_command_routes_to_the_handler() {
        let mut app = App::for_tests();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();

        app.set_chat_text("/mcp status");
        app.submit_message(tx.clone());

        assert!(!app.is_generating, "a status report arms no generation");
        let last = app.messages.last().unwrap();
        assert_eq!(last.role, "user");
        assert_eq!(last.content, "/mcp status");

        let said = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("status answer must arrive")
            .expect("status channel stays open");
        assert!(said.starts_with("[MCP]"), "{said}");
        assert!(said.contains("no MCP servers running"), "{said}");
    }

    /// Slash commands are local TUI verbs: they render in the transcript but
    /// must not enter persistent conversation memory, where they would be
    /// replayed to the model forever. Real prompts still do.
    #[tokio::test]
    async fn slash_commands_stay_out_of_conversation_memory() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel::<String>();

        let before = app.memory.get_context(100_000).len();
        app.set_chat_text("/mcp status");
        app.submit_message(tx.clone());
        assert_eq!(
            app.memory.get_context(100_000).len(),
            before,
            "a slash command must not grow memory"
        );

        app.set_chat_text("plain prompt for memory");
        app.submit_message(tx);
        let after = app.memory.get_context(100_000);
        assert_eq!(after.len(), before + 1);
        assert_eq!(after.last().unwrap().content, "plain prompt for memory");
    }

    /// I2-03: the list the agent posts belongs to the user as well — `/plan`
    /// pins it, `/plan clear` drops it. Seeded through the real gated path so
    /// the test also proves a plan costs no approval in `ask` mode.
    #[tokio::test]
    async fn plan_command_pins_and_clears_the_list_the_agent_posted() {
        let mut app = App::for_tests();
        let (prompts, _rx) = mpsc::unbounded_channel();
        let ctx = crate::agent_tools::ApprovalCtx {
            mode: crate::agent_tools::ApprovalMode::Ask,
            grants: app.agent_grants.clone(),
            prompts,
            checkpoints: app.checkpoints.clone(),
            turn: app.checkpoints.begin_turn(),
            command_timeout: crate::agent_tools::DEFAULT_COMMAND_TIMEOUT,
            plan: app.agent_plan.clone(),
            mcp: app.mcp.clone(),
            hooks: app.config.agent_hooks.clone(),
        };
        let call = xencode_providers_rs::ToolCall {
            id: "p1".to_string(),
            name: "update_plan".to_string(),
            arguments: serde_json::json!({"items": [
                {"text": "read the failing test", "status": "done"},
                {"text": "fix the parser", "status": "in_progress"},
                {"text": "re-run the suite", "status": "pending"},
            ]}),
        };
        let dir = std::env::temp_dir().join(format!("xencode-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let posted = crate::agent_tools::execute_tool_call_approved(
            &app.task_runtime,
            &dir,
            &call,
            &ctx,
            Some(&app.mcp),
        )
        .await;
        assert!(
            posted.starts_with("plan updated: 3 step(s), 1 done"),
            "{posted}"
        );

        app.handle_plan_command("/plan");
        assert!(app.plan_pinned);
        assert!(
            app.messages
                .last()
                .unwrap()
                .content
                .contains("Plan pinned: 1/3 steps done"),
            "{:?}",
            app.messages.last().unwrap().content
        );

        app.handle_plan_command("/plan");
        assert!(!app.plan_pinned);
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("Plan compact: 1/3 steps done"));

        app.handle_plan_command("/plan clear");
        assert!(crate::agent_tools::plan_items(&app.agent_plan).is_empty());
        assert!(!app.plan_pinned, "a cleared plan cannot stay pinned");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("Plan cleared"));

        // Clearing nothing says so instead of pretending to work.
        app.handle_plan_command("/plan clear");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("no plan to clear"));

        // A bad argument is usage, not a silent no-op.
        app.handle_plan_command("/plan expand");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("usage: /plan"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `/plan` is a viewer, not a request: it must never open a chat turn.
    #[tokio::test]
    async fn plan_command_does_not_start_a_generation() {
        let mut app = App::for_tests();
        app.set_chat_text("/plan");
        let (tx, _rx) = mpsc::unbounded_channel();
        app.submit_message(tx);
        assert!(!app.is_generating);
        assert_eq!(app.messages.last().unwrap().role, "system");
        assert!(app.messages.last().unwrap().content.contains("No plan yet"));
    }

    #[test]
    fn parse_llama_port_handles_common_urls() {
        assert_eq!(parse_llama_port("http://localhost:8080"), 8080);
        assert_eq!(parse_llama_port("http://127.0.0.1:8080/"), 8080);
        assert_eq!(parse_llama_port("https://host.example:11434/v1"), 11434);
        assert_eq!(parse_llama_port("http://localhost"), 8080);
        assert_eq!(parse_llama_port("8080"), 8080);
        assert_eq!(parse_llama_port(""), 8080);
    }

    /// Build `git status --porcelain -z` output: NUL after every field.
    fn porcelain_z(fields: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        for field in fields {
            out.extend_from_slice(field.as_bytes());
            out.push(0);
        }
        out
    }

    /// The keys must match what `scan_workspace` puts in `file_tree`: a path
    /// relative to the root, no prefix, forward slashes. This is the bug —
    /// keys used to be built as `.\src\main.rs` and never matched.
    #[test]
    fn porcelain_keys_match_the_file_tree_path_shape() {
        let status = parse_porcelain_z(&porcelain_z(&[" M src/main.rs", "?? notes.txt"]));

        assert_eq!(status.get("src/main.rs"), Some(&"M".to_string()));
        assert_eq!(status.get("notes.txt"), Some(&"??".to_string()));
        assert!(status.keys().all(|k| !k.contains('\\')), "{status:?}");
        assert!(status.keys().all(|k| !k.starts_with("./")), "{status:?}");
    }

    #[test]
    fn porcelain_trims_the_status_code() {
        let status = parse_porcelain_z(&porcelain_z(&[
            " M modified.rs",
            "A  added.rs",
            "MM both.rs",
            " D deleted.rs",
        ]));

        assert_eq!(status.get("modified.rs"), Some(&"M".to_string()));
        assert_eq!(status.get("added.rs"), Some(&"A".to_string()));
        assert_eq!(status.get("both.rs"), Some(&"MM".to_string()));
        assert_eq!(status.get("deleted.rs"), Some(&"D".to_string()));
    }

    /// A rename carries its original path as an extra field. If that field is
    /// not consumed, it is misread as the next entry and every entry after a
    /// rename is wrong.
    #[test]
    fn porcelain_handles_a_rename_without_desyncing() {
        let status = parse_porcelain_z(&porcelain_z(&[
            "R  new_name.rs",
            "old_name.rs", // the rename's original path
            " M after_the_rename.rs",
        ]));

        assert_eq!(status.get("new_name.rs"), Some(&"R".to_string()));
        // The entry after the rename must still be read correctly.
        assert_eq!(status.get("after_the_rename.rs"), Some(&"M".to_string()));
        // The original path is not a status entry of its own.
        assert!(!status.contains_key("old_name.rs"), "{status:?}");
        assert_eq!(status.len(), 2, "{status:?}");
    }

    #[test]
    fn porcelain_keeps_non_ascii_paths_verbatim() {
        // With -z these arrive unquoted and unescaped, unlike plain --porcelain
        // which would render this as "src/caf\303\251.rs" including the quotes.
        let status = parse_porcelain_z(&porcelain_z(&[" M src/café.rs", "?? 日本語.md"]));

        assert_eq!(status.get("src/café.rs"), Some(&"M".to_string()));
        assert_eq!(status.get("日本語.md"), Some(&"??".to_string()));
    }

    #[test]
    fn porcelain_handles_empty_and_truncated_input() {
        assert!(parse_porcelain_z(b"").is_empty());
        assert!(parse_porcelain_z(b"\0").is_empty());
        // Too short to be `XY <path>`.
        assert!(parse_porcelain_z(&porcelain_z(&["M"])).is_empty());
    }

    /// Pins the contract the parser targets: `scan_workspace` yields paths
    /// relative to the root with no `./` prefix, which is why the git path is
    /// used as the key verbatim.
    #[test]
    fn scan_workspace_paths_have_no_prefix() {
        let tmp = std::env::temp_dir().join(format!(
            "xencode-tui-gitkey-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src").join("main.rs"), "").unwrap();

        let entries = scan_workspace(&tmp, &ScanOptions::default()).unwrap();
        let paths: Vec<String> = entries
            .iter()
            .map(|e| e.path.display().to_string().replace('\\', "/"))
            .collect();

        std::fs::remove_dir_all(&tmp).ok();
        assert!(paths.contains(&"src/main.rs".to_string()), "{paths:?}");
    }

    fn watch_sets() -> (HashSet<String>, HashSet<String>) {
        (
            ["src/attached.rs".to_string()].into_iter().collect(),
            ["src/tracked.rs".to_string()].into_iter().collect(),
        )
    }

    #[test]
    fn first_output_line_picks_first_nonempty_line() {
        assert_eq!(
            first_output_line(b"[main abc1234] msg\n 2 files changed\n"),
            "[main abc1234] msg"
        );
        assert_eq!(first_output_line(b"\n\n  real  \nx"), "real");
        assert_eq!(first_output_line(b""), "(no output)");
        assert_eq!(first_output_line(b"   \n"), "(no output)");
    }

    #[test]
    fn text_entry_active_only_for_text_fields() {
        let mut app = super::App::for_tests();
        app.focus = FocusArea::ChatInput;
        assert!(!app.text_entry_active());
        app.focus = FocusArea::GitCommit;
        assert!(app.text_entry_active());
        app.focus = FocusArea::ByteBotPanel;
        assert!(app.text_entry_active());
        app.focus = FocusArea::Settings;
        app.settings_url_editing = false;
        assert!(!app.text_entry_active());
        app.settings_url_editing = true;
        assert!(app.text_entry_active());
        app.focus = FocusArea::CollaborationHub;
        app.collab_editing = false;
        assert!(!app.text_entry_active());
        app.collab_editing = true;
        assert!(app.text_entry_active());
    }

    #[test]
    fn collab_member_snapshots_replace_the_list_wholesale() {
        let mut app = super::App::for_tests();
        app.collab_members = vec![("ghost".into(), "editor".into(), "connected".into())];
        app.apply_collab_token(
            r#"members:[{"username":"alice","role":"admin"},{"username":"bob","role":"viewer"}]"#,
        );
        assert_eq!(app.collab_members.len(), 2);
        assert_eq!(
            app.collab_members[0],
            (
                "alice".to_string(),
                "admin".to_string(),
                "connected".to_string()
            )
        );
        assert_eq!(app.collab_members[1].1, "viewer");

        app.apply_collab_token("session:xencode-77");
        assert_eq!(app.collab_session_id, "xencode-77");

        app.apply_collab_token("status:connected");
        assert_eq!(app.collab_sync_status, "connected");
        app.apply_collab_token("status:disconnected");
        assert!(!app.collab_session_active);
        assert_eq!(app.collab_sync_status, "disconnected");

        app.apply_collab_token("error:bad_token: invalid or expired token");
        assert_eq!(app.collab_error, "bad_token: invalid or expired token");

        // Garbage must not clobber the current list — it reports honestly.
        app.apply_collab_token("members:not-json");
        assert_eq!(app.collab_members.len(), 2);
        assert_eq!(app.collab_error, "malformed members list from server");
    }

    #[test]
    fn watch_warning_formats_per_kind() {
        assert!(format_watch_warning("src/a.rs", "removed").contains("removed from disk"));
        assert!(format_watch_warning("src/a.rs", "created").contains("created on disk"));
        assert!(format_watch_warning("src/a.rs", "modified").contains("changed on disk"));
        // Unknown kinds fall back to the generic changed text.
        assert!(format_watch_warning("src/a.rs", "renamed").contains("changed on disk"));
    }

    #[test]
    fn watch_warning_fires_for_context_files_only() {
        let (attached, tracked) = watch_sets();
        let deps: Vec<String> = vec!["src/dep.rs".to_string()];
        assert!(watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &deps,
            None
        )
        .unwrap()
        .contains("Dependents to re-check: src/dep.rs"));
        assert!(watch_warning_for(
            "src/open.rs",
            "modified",
            &attached,
            Some("src/open.rs"),
            &tracked,
            &[],
            None
        )
        .is_some());
        assert!(watch_warning_for(
            "src/tracked.rs",
            "removed",
            &attached,
            None,
            &tracked,
            &[],
            None
        )
        .unwrap()
        .contains("removed from disk"));
        // Unrelated file → no warning, even with dependents.
        assert!(watch_warning_for(
            "src/other.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &deps,
            None
        )
        .is_none());
    }

    #[test]
    fn watch_warning_dedupes_and_truncates() {
        let (attached, tracked) = watch_sets();
        let warning = watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &[],
            None,
        )
        .unwrap();
        // Same path already the last system message → suppressed.
        assert!(watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &[],
            Some(&warning)
        )
        .is_none());
        // Long dependent lists show 5 names plus an overflow count.
        let many: Vec<String> = (0..7).map(|i| format!("src/d{i}.rs")).collect();
        let long = watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &many,
            None,
        )
        .unwrap();
        assert!(long.contains("(+2 more)"), "{long}");
        assert!(!long.contains("src/d5.rs"), "{long}");
    }

    #[test]
    fn watch_dedup_matches_exact_head_not_substrings() {
        let (attached, tracked) = watch_sets();
        // A dependents list mentioning a SIBLING path must not suppress this
        // path's warning: suppression keys on the exact warning head.
        let sibling_deps = vec!["src/attached.rs.bak".to_string()];
        assert!(watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &sibling_deps,
            Some("earlier note about src/attached.rs.bak here"),
        )
        .is_some());
        // Same kind + path already shown → suppressed…
        let warning = watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &[],
            None,
        )
        .unwrap();
        assert!(watch_warning_for(
            "src/attached.rs",
            "modified",
            &attached,
            None,
            &tracked,
            &[],
            Some(&warning),
        )
        .is_none());
        // …but a kind change re-warns.
        assert!(watch_warning_for(
            "src/attached.rs",
            "removed",
            &attached,
            None,
            &tracked,
            &[],
            Some(&warning),
        )
        .is_some());
    }

    fn advise_of(kind: xencode_context_rs::AdviceKind, file: &str) -> xencode_context_rs::Advice {
        xencode_context_rs::Advice {
            file: file.to_string(),
            kind,
            message: format!("{file} — test finding"),
        }
    }

    #[test]
    fn advise_report_summarizes_counts() {
        use xencode_context_rs::AdviceKind;
        let all = vec![
            advise_of(AdviceKind::BrokenImport, "src/a.rs"),
            advise_of(AdviceKind::Cycle, "src/a.rs"),
            advise_of(AdviceKind::Cycle, "src/b.rs"),
            advise_of(AdviceKind::Orphan, "src/z.rs"),
            advise_of(AdviceKind::AffectedDependent, "src/c.rs"),
        ];
        let report = format_advise_report(&all, None);
        assert_eq!(report.len(), 6, "{report:?}");
        assert!(
            report[0].contains("5 findings")
                && report[0].contains("1 broken import")
                && report[0].contains("2 cycles")
                && report[0].contains("1 orphan")
                && report[0].contains("1 affected dependent"),
            "{}",
            report[0]
        );
        // A filter narrows the header to what is shown, naming the total.
        let filtered = format_advise_report(&all, Some("src/a.rs"));
        assert!(filtered[0].contains("2 findings"), "{}", filtered[0]);
        assert!(filtered[0].contains("5 total"), "{}", filtered[0]);
    }

    #[test]
    fn advise_report_filters_caps_and_handles_empty() {
        use xencode_context_rs::AdviceKind;
        let all = vec![
            advise_of(AdviceKind::Hub, "src/router.rs"),
            advise_of(AdviceKind::Orphan, "src/old.rs"),
        ];
        let filtered = format_advise_report(&all, Some("router"));
        assert_eq!(filtered.len(), 2, "{filtered:?}");
        assert!(filtered[1].contains("src/router.rs"), "{filtered:?}");

        let missed = format_advise_report(&all, Some("nothing-matches"));
        assert_eq!(missed.len(), 1);
        assert!(missed[0].contains("No findings matching"), "{}", missed[0]);

        let empty = format_advise_report(&[], None);
        assert_eq!(empty.len(), 1);
        assert!(empty[0].contains("No findings"), "{}", empty[0]);

        // Overflow past the cap collapses into a narrow-hint line.
        let many: Vec<xencode_context_rs::Advice> = (0..(super::ADVISE_LINE_CAP + 3))
            .map(|i| advise_of(AdviceKind::Orphan, &format!("src/f{i:03}.rs")))
            .collect();
        let capped = format_advise_report(&many, None);
        assert_eq!(capped.len(), super::ADVISE_LINE_CAP + 2, "{capped:?}");
        assert!(capped.last().unwrap().contains("+3 more"), "{capped:?}");
    }

    fn image_test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-tui-img-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_png() -> Vec<u8> {
        let mut v = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        v.extend_from_slice(&13u32.to_be_bytes());
        v.extend_from_slice(b"IHDR");
        v.extend_from_slice(&2u32.to_be_bytes());
        v.extend_from_slice(&2u32.to_be_bytes());
        v.extend_from_slice(&[8, 2, 0, 0, 0]);
        v
    }

    #[test]
    fn encode_attached_image_produces_data_url() {
        let dir = image_test_dir("ok");
        let path = dir.join("shot.png");
        let bytes = tiny_png();
        std::fs::write(&path, &bytes).unwrap();
        let url = super::encode_attached_image(path.to_str().unwrap()).unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn encode_attached_image_reports_skips() {
        let dir = image_test_dir("skip");
        let fake = dir.join("fake.png");
        std::fs::write(&fake, b"not an image").unwrap();
        let err = super::encode_attached_image(fake.to_str().unwrap()).unwrap_err();
        assert_eq!(err, "not a recognized image");
        let missing =
            super::encode_attached_image(dir.join("gone.png").to_str().unwrap()).unwrap_err();
        assert!(missing.starts_with("cannot read file:"), "{missing}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn attach_images_patches_final_user_turn() {
        use xencode_providers_rs::{ChatMessage, ContentPart, MessageContent};
        let mut messages = vec![
            ChatMessage::text("system", "sys"),
            ChatMessage::text("user", "look at this"),
        ];
        let patched = super::attach_images_to_last_message(
            &mut messages,
            vec!["data:image/png;base64,AAAA".to_string()],
        );
        assert!(patched);
        assert_eq!(
            messages[1].content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "look at this".to_string()
                },
                ContentPart::ImageUrl {
                    image_url: xencode_providers_rs::ImageUrlPart {
                        url: "data:image/png;base64,AAAA".to_string(),
                        detail: None,
                    },
                },
            ])
        );
        // Earlier turns are untouched.
        assert_eq!(messages[0].text_content(), "sys");
    }

    #[test]
    fn attach_images_refuses_non_user_tail_and_empty_urls() {
        use xencode_providers_rs::ChatMessage;
        let mut messages = vec![ChatMessage::text("assistant", "done")];
        assert!(!super::attach_images_to_last_message(
            &mut messages,
            vec!["data:image/png;base64,AAAA".to_string()],
        ));
        assert_eq!(messages[0].text_content(), "done");

        let mut empty: Vec<ChatMessage> = Vec::new();
        assert!(!super::attach_images_to_last_message(&mut empty, vec![]));
    }

    fn doc_of(text: &str) -> xencode_context_rs::DocText {
        xencode_context_rs::DocText {
            path: "paper.pdf".to_string(),
            kind: xencode_context_rs::DocKind::Pdf,
            text: text.to_string(),
            truncated: false,
            bytes: 100,
        }
    }

    #[test]
    fn doc_block_inlines_text_and_names_skips() {
        let text = super::doc_attach_block("paper.pdf", &Ok(doc_of("Hello paper")));
        assert_eq!(text, "<file path=\"paper.pdf\">\nHello paper\n</file>\n\n");

        let empty = super::doc_attach_block("scan.pdf", &Ok(doc_of("   \n ")));
        assert!(empty.contains("no extractable text"), "{empty}");

        let failed: Result<xencode_context_rs::DocText, String> = Err("bogus".to_string());
        let err = super::doc_attach_block("bad.pdf", &failed);
        assert!(err.contains("(document not parsed: bogus)"), "{err}");
    }

    #[test]
    fn parse_attached_document_reports_garbage_and_missing() {
        let dir = image_test_dir("doc");
        let fake = dir.join("fake.pdf");
        std::fs::write(&fake, b"not a pdf at all").unwrap();
        let err = super::parse_attached_document(fake.to_str().unwrap()).unwrap_err();
        assert!(err.contains("pdf parse failed"), "{err}");

        let missing =
            super::parse_attached_document(dir.join("gone.pdf").to_str().unwrap()).unwrap_err();
        assert!(missing.starts_with("cannot read file:"), "{missing}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn text_attachment_inlines_readable_and_notes_unreadable() {
        let dir = image_test_dir("txt");
        let ok = dir.join("note.txt");
        std::fs::write(&ok, "hello").unwrap();
        let mut block = String::new();
        super::append_text_attachment(&mut block, ok.to_str().unwrap());
        assert_eq!(
            block,
            format!("<file path=\"{}\">\nhello\n</file>\n\n", ok.display())
        );

        // Binary content fails read_to_string → visible note, not silence.
        let bin = dir.join("blob.bin");
        std::fs::write(&bin, [0xFF, 0xFE, 0x00]).unwrap();
        let mut block2 = String::new();
        super::append_text_attachment(&mut block2, bin.to_str().unwrap());
        assert!(block2.contains("(attachment not sent:"), "{block2}");
        assert!(block2.contains(&bin.display().to_string()), "{block2}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression for E2-01, rewritten for I2-04: Enter in the ByteBot panel
    /// used to overwrite the typed command with the last history entry, and
    /// the run itself was a script. Now it arms the real tool loop — the task
    /// stays on screen, and no step row exists until a call does.
    #[test]
    fn bytebot_arms_a_real_run_from_the_typed_task() {
        let mut app = super::App::for_tests();
        app.bytebot_history = vec!["previous command".to_string()];
        app.bytebot_command = "fix flaky tests".to_string();
        app.bytebot_cursor = app.bytebot_command.len();
        // The fallback chain rides into the run (I4-01): the primary model
        // first, then these alternates, in this order.
        app.config.agent_fallback_models = vec![
            "qwen2.5:14b".to_string(),
            "gemini:gemini-2.0-flash".to_string(),
        ];

        let run = app
            .arm_bytebot("fix flaky tests")
            .expect("a typed task arms a run");
        assert!(app.bytebot_running);
        assert_eq!(
            app.bytebot_command, "fix flaky tests",
            "the task stays visible while it runs"
        );
        assert!(
            app.bytebot_steps.is_empty(),
            "no step exists before a call does: {:?}",
            app.bytebot_steps
        );
        assert_eq!(run.sink, LoopSink::ByteBot);
        assert_eq!(run.max_rounds, app.config.agent_max_rounds.clamp(1, 64));
        assert_eq!(
            run.fallback_models,
            vec![
                "qwen2.5:14b".to_string(),
                "gemini:gemini-2.0-flash".to_string()
            ],
            "the configured alternates ride along in order"
        );
        // The brief and the tool vocabulary ride along; the model is not
        // asked to guess that it may edit files.
        let system = run.context_messages.first().expect("system turn");
        let xencode_providers_rs::MessageContent::Text(system_text) = &system.content else {
            panic!("the system turn is text");
        };
        assert!(system_text.contains("update_plan"), "{system_text}");
        let last = run.context_messages.last().expect("task turn");
        let xencode_providers_rs::MessageContent::Text(task_text) = &last.content else {
            panic!("the task turn is text");
        };
        assert!(task_text.contains("fix flaky tests"), "{task_text}");
        assert!(task_text.contains("Delegated task"), "{task_text}");
        assert_eq!(
            app.bytebot_history,
            vec![
                "previous command".to_string(),
                "fix flaky tests".to_string()
            ],
            "history is remembered, not recalled"
        );

        // A second task while the first runs is refused, not queued blindly.
        assert!(app.arm_bytebot("and again").is_none());
        assert!(app
            .toasts
            .iter()
            .any(|toast| toast.message.contains("already working")));
    }

    /// I4-01: what stops the chain. A candidate that streamed anything keeps
    /// the turn, and so does an error in our own decoder — the next model
    /// would hit it just the same.
    #[test]
    fn only_a_clean_fallback_eligible_failure_advances_the_chain() {
        use xencode_providers_rs::ProviderError;

        let provider_down = ProviderError::api("OpenRouter", 503u16, "overloaded");
        assert!(
            super::should_advance_fallback(&provider_down, false),
            "a clean provider failure should move to the alternate"
        );
        assert!(
            !super::should_advance_fallback(&provider_down, true),
            "tokens already on screen fix the model in place"
        );
        let bad_decode = ProviderError::Parse("OpenRouter: empty response".to_string());
        assert!(
            !super::should_advance_fallback(&bad_decode, false),
            "our own parse failure is not the provider's fault; the chain stops"
        );
    }

    /// `/bytebot <task>` from the chat is the same run, and an argument-less
    /// one is usage rather than a silent no-op.
    #[tokio::test]
    async fn bytebot_command_arms_the_panel_from_chat() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.set_chat_text("/bytebot");
        app.submit_message(tx.clone());
        assert!(!app.bytebot_running);
        assert!(!app.is_generating, "delegating is not a chat turn");
        assert!(app
            .messages
            .last()
            .unwrap()
            .content
            .contains("usage: /bytebot"));
    }

    #[test]
    fn bytebot_steps_are_the_calls_and_progress_is_their_outcome() {
        let mut app = App::for_tests();
        app.bytebot_running = true;

        app.bytebot_event("call:read_file src/app.rs");
        app.bytebot_event("done:done");
        assert_eq!(
            app.bytebot_steps,
            vec![("read_file src/app.rs".to_string(), "done".to_string())]
        );
        assert_eq!(app.bytebot_progress, 1.0);

        app.bytebot_event("call:edit_file src/app.rs");
        assert_eq!(
            app.bytebot_progress, 0.5,
            "an in-flight call counts against the total, not for it"
        );
        app.bytebot_event("done:denied");
        assert_eq!(app.bytebot_steps[1].1, "denied");

        app.bytebot_event("log:renamed the helper");
        assert_eq!(app.bytebot_log.last().unwrap(), "renamed the helper");

        // A provider failure is reported, and leaves the open call failed
        // rather than spinning forever.
        app.bytebot_event("call:run_command cargo test");
        app.bytebot_event("err:connection refused");
        assert_eq!(app.bytebot_steps[2].1, "failed");
        assert!(app
            .bytebot_log
            .last()
            .unwrap()
            .contains("connection refused"));
    }

    #[tokio::test]
    async fn multiline_submit_keeps_newlines_and_clears_box() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        // "/init abort" is intercepted before any provider call, so the
        // user-message capture can be asserted without spawning work.
        app.chat_input.insert_str("/init abort");
        app.chat_input.insert_newline();
        app.chat_input.insert_str("second line");
        app.submit_message(tx);
        let last = app.messages.last().expect("message pushed");
        assert_eq!(last.content, "/init abort\nsecond line");
        assert_eq!(app.chat_input.lines().join("\n"), "");
    }

    #[test]
    fn slash_completion_extends_unique_prefixes_only() {
        use super::complete_slash_token;
        assert_eq!(complete_slash_token("/ini").as_deref(), Some("/init"));
        assert_eq!(complete_slash_token("/c").as_deref(), Some("/ctx"));
        assert_eq!(complete_slash_token("/b").as_deref(), Some("/bytebot"));
        assert_eq!(complete_slash_token("/init").as_deref(), None); // complete
        assert_eq!(complete_slash_token("/x").as_deref(), None); // no match
        assert_eq!(complete_slash_token("/").as_deref(), None); // listing case
        assert_eq!(complete_slash_token("hello").as_deref(), None);
    }

    #[tokio::test]
    async fn history_recall_walks_entries_and_restores_draft() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.chat_input.insert_str("/init abort");
        app.submit_message(tx.clone());
        app.chat_input.insert_str("/ctx status");
        app.submit_message(tx.clone());
        app.chat_input.insert_str("/ctx status");
        app.submit_message(tx.clone()); // adjacent duplicate → not stored twice
        app.chat_input.insert_str("current draft");

        app.recall_history(-1);
        assert_eq!(app.chat_input.lines().join("\n"), "/ctx status");
        app.recall_history(-1);
        assert_eq!(app.chat_input.lines().join("\n"), "/init abort");
        app.recall_history(-1); // clamps at oldest
        assert_eq!(app.chat_input.lines().join("\n"), "/init abort");
        app.recall_history(1);
        assert_eq!(app.chat_input.lines().join("\n"), "/ctx status");
        app.recall_history(1); // past newest → stashed draft returns
        assert_eq!(app.chat_input.lines().join("\n"), "current draft");
        assert_eq!(
            app.input_history.len(),
            2,
            "dedup keeps one copy per prompt"
        );
    }

    /// Receive side of the `[TASKS]` channel protocol (the send side is
    /// covered in `keymap.rs`): malformed bodies are silent no-ops,
    /// `stop|id` and `rm|id` mutate the shared registry off-thread.
    #[tokio::test]
    async fn tasks_command_mutates_registry_and_ignores_junk() {
        let mut app = App::for_tests();
        app.task_runtime
            .lock()
            .await
            .start("sleeper", "sleep 30")
            .await
            .unwrap();
        for junk in ["", "stop", "stop|x", "bogus|1"] {
            app.handle_tasks_command(junk);
        }
        tokio::task::yield_now().await;
        {
            let m = app.task_runtime.lock().await;
            assert_eq!(m.list().len(), 1, "junk bodies must not touch the registry");
            assert_eq!(m.list()[0].status, TaskStatus::Running);
        }
        app.handle_tasks_command("stop|1");
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if !matches!(
                app.task_runtime.lock().await.list()[0].status,
                TaskStatus::Running
            ) {
                break;
            }
        }
        assert_eq!(
            app.task_runtime.lock().await.list()[0].status,
            TaskStatus::Killed
        );
        app.handle_tasks_command("rm|1");
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if app.task_runtime.lock().await.list().is_empty() {
                break;
            }
        }
        assert!(app.task_runtime.lock().await.list().is_empty());
    }

    #[test]
    fn live_refresh_snapshot_updates_edited_rs_files() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "xencode-liverefresh-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\n").unwrap();
        std::fs::write(root.join("src/a.rs"), "use crate::b::bee;\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "pub fn bee() {}\n").unwrap();
        xencode_context_rs::init_project(
            &root,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            |_| {},
        )
        .expect("init");
        let deps = |root: &std::path::Path| -> Vec<xencode_context_rs::DepEdge> {
            xencode_context_rs::read_json(&xencode_context_rs::deps_json_path(
                &root.join(xencode_context_rs::XENCODE_DIR),
            ))
            .unwrap()
        };
        assert!(deps(&root)
            .iter()
            .any(|e| e.from == "src/a.rs" && e.to == "src/b.rs"));

        // Edited file: edge vanishes from the on-disk snapshot.
        std::fs::write(root.join("src/a.rs"), "pub fn ay() {}\n").unwrap();
        assert!(live_refresh_snapshot(&root, "modified", "src/a.rs"));
        assert!(!deps(&root).iter().any(|e| e.from == "src/a.rs"));

        // Gates: wrong kinds and non-Rust paths never touch the snapshot.
        assert!(!live_refresh_snapshot(&root, "created", "src/a.rs"));
        assert!(!live_refresh_snapshot(&root, "modified", "src/lib.txt"));
        // Unknown (never-indexed) path → refresh is a no-op → false.
        assert!(!live_refresh_snapshot(&root, "modified", "src/ghost.rs"));

        // Removed file: its symbol record drops out of the snapshot.
        std::fs::remove_file(root.join("src/b.rs")).unwrap();
        assert!(live_refresh_snapshot(&root, "removed", "src/b.rs"));
        let symbols: std::collections::BTreeMap<String, xencode_context_rs::PerFileSymbols> =
            xencode_context_rs::read_json(&xencode_context_rs::symbols_json_path(
                &root.join(xencode_context_rs::XENCODE_DIR),
            ))
            .unwrap();
        assert!(!symbols.contains_key("src/b.rs"));

        std::fs::remove_dir_all(&root).unwrap();
    }

    // I3-03 /spawn tests ───────────────────────────────────────────

    #[test]
    fn spawn_parsing_treats_a_hash_lead_trail_as_an_optional_branch() {
        let (task, branch) = App::parse_spawn("add tests #feat");
        assert_eq!(task, "add tests");
        assert_eq!(branch, Some("#feat"));
        // No # token: everything is the task, branches stay generated.
        let (task, branch) = App::parse_spawn("run the whole suite");
        assert_eq!(task, "run the whole suite");
        assert_eq!(branch, None);
        // A lone #branch leaves no task — the handler will say usage.
        let (task, branch) = App::parse_spawn("#bump");
        assert_eq!(task, "");
        assert_eq!(branch, Some("#bump"));
    }

    #[test]
    fn spawn_events_track_a_run_and_post_the_finish_report() {
        let mut app = App::for_tests();
        app.spawns.push(SpawnRecord {
            id: 1,
            branch: "xencode/spawn-1".to_string(),
            path: std::path::PathBuf::from("/tmp/xencode-spawn-1"),
            task: "add tests for spawn".to_string(),
            running: true,
            failed: false,
            steps: Vec::new(),
        });
        app.spawn_event(1, "call:read_file src/app.rs");
        app.spawn_event(1, "done:done");
        app.spawn_event(1, "call:write_file src/app.rs");
        app.spawn_event(1, "done:done");
        app.spawn_event(1, "finish:done! the tests pass");

        assert!(!app.spawns[0].running);
        assert!(!app.spawns[0].failed);
        assert_eq!(app.spawns[0].steps.len(), 2);
        // The report lands in the chat: a system line for where/what, then an
        // assistant message with the agent's final answer.
        let last = app.messages.last().unwrap();
        assert_eq!(last.role, "assistant");
        assert_eq!(
            last.content,
            "(spawn #1 · add tests for spawn)\ndone! the tests pass"
        );
        let sys = app
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "system")
            .unwrap();
        assert!(sys.content.contains("spawn #1 done"), "{}", sys.content);
        assert!(
            sys.content.contains("2/2 call(s) completed"),
            "{}",
            sys.content
        );
        assert!(
            sys.content.contains("branch `xencode/spawn-1`"),
            "{}",
            sys.content
        );
    }

    #[test]
    fn spawn_events_surface_a_real_failure_without_inventing_an_answer() {
        let mut app = App::for_tests();
        app.spawns.push(SpawnRecord {
            id: 2,
            branch: "xencode/spawn-2".to_string(),
            path: std::path::PathBuf::from("/tmp/xencode-spawn-2"),
            task: "run the tests".to_string(),
            running: true,
            failed: false,
            steps: Vec::new(),
        });
        app.spawn_event(2, "call:run_command cargo test");
        app.spawn_event(2, "err:connection refused");
        app.spawn_event(2, "finish:");

        assert!(app.spawns[0].failed);
        assert!(!app.spawns[0].running);
        // The running step is marked failed, not left hanging as running.
        assert_eq!(app.spawns[0].steps[0].1, "failed");
        // No invented text: err: reported the truth and finish was empty, so
        // no assistant message is pushed after an empty final text.
        assert!(!app.messages.iter().any(|m| m.role == "assistant"));
        assert!(app
            .messages
            .iter()
            .rev()
            .any(|m| m.role == "system"
                && m.content.contains("spawn #2 failed — connection refused")));
    }

    #[test]
    fn spawn_status_lists_every_registered_subagent() {
        let mut app = App::for_tests();
        app.spawns.push(SpawnRecord {
            id: 1,
            branch: "xencode/spawn-1".to_string(),
            path: std::path::PathBuf::from("/tmp/xencode-spawn-1"),
            task: "still working".to_string(),
            running: true,
            failed: false,
            steps: vec![("read_file src/app.rs".to_string(), "running".to_string())],
        });
        app.spawns.push(SpawnRecord {
            id: 2,
            branch: "xencode/spawn-2".to_string(),
            path: std::path::PathBuf::from("/tmp/xencode-spawn-2"),
            task: "finished work".to_string(),
            running: false,
            failed: false,
            steps: vec![("write_file src/lib.rs".to_string(), "done".to_string())],
        });
        let before = app.messages.len();
        app.handle_spawn_command("/spawn status", mpsc::unbounded_channel().0);
        let lines: Vec<&str> = app.messages[before..]
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("#1 running"), "{}", lines[0]);
        assert!(lines[0].contains("`xencode/spawn-1`"), "{}", lines[0]);
        assert!(lines[1].contains("#2 done    "), "{}", lines[1]);
        assert!(lines[1].contains("1 call(s)"), "{}", lines[1]);
    }

    #[test]
    fn spawn_worktree_creates_a_sibling_branch_worktree() {
        let tmp = std::env::temp_dir().join(format!("xencode-spawn-wt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("proj");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-b", "main"]);
        std::fs::write(repo.join("f.txt"), "hi").unwrap();
        git(&repo, &["add", "f.txt"]);
        git(
            &repo,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-m",
                "init",
            ],
        );

        // A named branch lands in `proj-spawn-<id>-<branch>` as a sibling.
        let named = super::spawn_worktree(&repo, 1, "feat").unwrap();
        assert_eq!(named, tmp.join("proj-spawn-1-feat"));
        let list = xencode_context_rs::worktree_list(&repo).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[1].branch.as_deref(), Some("feat"));
        // The committed content is checked out in the worktree.
        assert_eq!(std::fs::read_to_string(named.join("f.txt")).unwrap(), "hi");

        // The generated default branch gets a clean leaf without a suffix.
        let default = super::spawn_worktree(&repo, 2, "xencode/spawn-2").unwrap();
        assert_eq!(default, tmp.join("proj-spawn-2"));
        assert_eq!(
            xencode_context_rs::worktree_list(&repo).unwrap()[2]
                .branch
                .as_deref(),
            Some("xencode/spawn-2")
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The security panel's data has to come from the scanner, so a fixture
    /// tree with one known credential line must produce that finding and
    /// nothing from the scripted list it replaces (`src/config.py`,
    /// "142 tests passed").
    #[tokio::test]
    async fn security_scan_streams_real_findings() {
        let dir = std::env::temp_dir().join(format!(
            "xcode-sec-scan-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("src/db.rs"),
            "fn connect() {\n    let api_key = \"supersecretvalue123\";\n}\n",
        )
        .unwrap();
        // Named as a secret by the walker, and scannable if it were opened.
        std::fs::write(
            dir.join(".env"),
            "AWS_SECRET_ACCESS_KEY=aklsdjflaksdjflkjasdflkjas\n",
        )
        .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        super::run_security_scan(dir.clone(), tx).await;
        let mut messages = Vec::new();
        while let Ok(m) = rx.try_recv() {
            messages.push(m);
        }
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            messages.iter().any(
                |m| m.starts_with("[SECURITY]finding:Critical|hardcoded-secret|")
                    && m.contains("src/db.rs")
            ),
            "no real finding in {messages:?}"
        );
        // The secret file is named, not opened: one classed finding, no line.
        let secret_lines: Vec<&String> = messages.iter().filter(|m| m.contains(".env")).collect();
        assert_eq!(secret_lines.len(), 1, "{secret_lines:?}");
        assert!(secret_lines[0].starts_with("[SECURITY]finding:Medium|secret-file|.env|"));
        // Totals ride on the done line so the summary survives the display cap.
        let done = messages
            .iter()
            .find(|m| m.starts_with("[SECURITY]done:"))
            .expect("no done line");
        assert!(done.ends_with("|1,0,1,0"), "{done}");
        assert!(!messages.iter().any(|m| m.contains("config.py")));
        assert!(!messages.iter().any(|m| m.contains("tests passed")));
    }

    /// J-04: every row this panel shows comes out of the walk. The five files
    /// it used to list (`src/app.py`, `src/components.tsx`, …) do not exist in
    /// any workspace, and neither did the six-row "supported" table.
    #[tokio::test]
    async fn language_scan_reports_the_walk_not_a_script() {
        let dir = temp_dir("lang-scan");
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        // The blank line is not a line of code, so this file is 2, not 3.
        std::fs::write(dir.join("src/b.rs"), "fn c() {}\n\nfn d() {}\n").unwrap();
        std::fs::write(dir.join("notes.md"), "Title\nsome text\n").unwrap();
        // Named as a secret: listed, never read, so it adds a file and no lines.
        std::fs::write(
            dir.join(".env"),
            "AWS_SECRET_ACCESS_KEY=aklsdjflaksdjflkjasdflkjas\n",
        )
        .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        super::run_language_scan(dir.clone(), tx).await;
        let _ = std::fs::remove_dir_all(&dir);
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut done = false;
        while let Ok(token) = rx.try_recv() {
            if let Some(json) = token.strip_prefix("[LANG]row:") {
                rows.push(serde_json::from_str(json).unwrap());
            } else if let Some(note) = token.strip_prefix("[LANG]note:") {
                notes.push(note.to_string());
            } else if token == "[LANG]done" {
                done = true;
            }
        }
        assert!(done, "the scan never finished");

        let rust = rows
            .iter()
            .find(|r| r["language"] == "rust")
            .expect("no rust row");
        assert_eq!(rust["files"], 2, "{rows:?}");
        assert_eq!(rust["lines"], 4, "{rows:?}");
        assert_eq!(rust["share"], 66.7, "{rows:?}");
        assert_eq!(
            rows.first().map(|r| &r["language"]),
            Some(&serde_json::json!("rust")),
            "the biggest language comes first: {rows:?}"
        );
        assert!(
            rows.iter()
                .any(|r| r["language"] == "markdown" && r["files"] == 1 && r["lines"] == 2),
            "{rows:?}"
        );
        assert!(
            rows.iter().all(|r| r["language"] != "python"),
            "a language with no files must not appear: {rows:?}"
        );
        // Files the walk could not count are said, not hidden in the totals.
        assert!(
            notes
                .iter()
                .any(|n| n.contains("listed as secret") && n.contains("no lines")),
            "{notes:?}"
        );
        assert!(
            notes.iter().any(|n| n.starts_with("4 files · 6 lines")),
            "{notes:?}"
        );
        for canned in [
            "src/app.py",
            "components.tsx",
            "templates/index.html",
            "98.7",
            "99.2",
        ] {
            assert!(!format!("{rows:?}").contains(canned), "{canned}");
        }
    }

    /// A translation request with nothing to translate spends no provider call.
    #[test]
    fn translating_nothing_asks_nothing() {
        let mut app = App::for_tests();
        let (tx, mut rx) = mpsc::unbounded_channel();
        app.translate_text(tx);
        assert!(!app.lang_busy);
        assert!(app.lang_translate_output.contains("Nothing to translate"));
        assert!(rx.try_recv().is_err(), "no request should have started");

        // Editing is a mode: letters go to the selected field, not to commands.
        app.cycle_lang_field();
        assert_eq!(app.lang_editing, Some(crate::focus::LangField::Input));
        for c in "bonjour".chars() {
            app.lang_char(c);
        }
        assert_eq!(app.lang_translate_input, "bonjour");
        // Backspace deletes a character, so it has to be typed before the
        // deletion and asserted after it — not compared with itself.
        app.lang_translate_input.push('!');
        app.lang_backspace();
        assert_eq!(app.lang_translate_input, "bonjour");
        app.cycle_lang_field();
        assert_eq!(app.lang_editing, Some(crate::focus::LangField::Source));
        app.lang_char('!');
        assert_eq!(app.lang_translate_source, "auto!");
        app.lang_editing = None;
        app.lang_char('?');
        assert_eq!(app.lang_translate_source, "auto!");
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xcode-{}-{}", tag, nanos));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Run the real indexer over a temp workspace with three Rust files:
    /// `two.rs` declares three things, `one.rs` one, `quiet.rs` nothing.
    fn indexed_root(tag: &str) -> std::path::PathBuf {
        let root = temp_dir(tag);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/two.rs"),
            "pub struct Two {}\npub struct Dos {}\npub fn two() {}\n",
        )
        .unwrap();
        std::fs::write(root.join("src/one.rs"), "pub fn one() {}\n").unwrap();
        std::fs::write(root.join("src/quiet.rs"), "pub const MODE: u32 = 3;\n").unwrap();
        init_project(
            &root,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            |_| {},
        )
        .expect("init");
        root
    }

    /// The queue is the index's own list: files that declare something, most
    /// declarations first, ties by path. A file the extractor found nothing in
    /// is not a lesson.
    #[test]
    fn learning_lessons_come_from_the_index_in_a_stable_order() {
        let root = indexed_root("learn-queue");
        let lessons = learning_lessons(&root).expect("a queue");
        assert_eq!(
            lessons
                .iter()
                .map(|(path, _)| path.as_str())
                .collect::<Vec<_>>(),
            vec!["src/two.rs", "src/one.rs"],
        );
        assert!(lessons[0].1.contains(&"struct Two".to_string()));
        assert!(lessons[0].1.contains(&"fn two".to_string()));

        // No index: the panel gets a reason to show, not a lesson to fake.
        let empty = temp_dir("learn-none");
        assert_eq!(
            learning_lessons(&empty),
            Err("No project index — run /init first, then Enter again.".to_string())
        );
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&empty).unwrap();
    }

    /// What the panel prints for a lesson is the file's own text and the
    /// declarations the index recorded — including what it had to leave out.
    #[test]
    fn learn_show_prints_the_real_file_and_what_it_declares() {
        let root = temp_dir("learn-show");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let long = format!("pub struct Big {{}}\n{}", "fn filler() {{}}\n".repeat(400));
        std::fs::write(root.join("src/big.rs"), &long).unwrap();
        init_project(
            &root,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            |_| {},
        )
        .expect("init");

        let mut app = App::for_tests();
        app.learn_root = root.clone();
        app.learn_lessons = learning_lessons(&root).unwrap();
        assert!(app.learn_show(1));
        assert_eq!(app.learn_lesson_title, "src/big.rs");
        assert!(app.learn_code_example.contains("pub struct Big"));
        assert!(app.learn_content[0].contains("declaration(s) the index found in src/big.rs"));
        // The file is bigger than the cap, and the panel says how much it sent.
        let cap_note = &app.learn_content[2];
        assert!(cap_note.starts_with("First "), "{cap_note}");
        assert!(
            cap_note.contains("bytes sent — the file is longer"),
            "{cap_note}"
        );
        assert!(!app.learn_busy, "showing a file asks nothing of a provider");

        // A lesson past the end of the queue changes nothing.
        app.learn_status.clear();
        assert!(!app.learn_show(9));
        assert!(app.learn_status.is_empty());

        // The index names a file that has since gone: said out loud, and the
        // previous lesson's text is not left standing as if it were this one.
        std::fs::remove_file(root.join("src/big.rs")).unwrap();
        assert!(!app.learn_show(1));
        assert!(
            app.learn_status.starts_with("The index names src/big.rs"),
            "{}",
            app.learn_status
        );
        assert!(app.learn_content.is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The answer key is the model's, and grading is against it — not against
    /// whichever option happens to be first, as it used to be.
    #[test]
    fn learn_quiz_grades_against_the_models_key() {
        let mut app = App::for_tests();
        app.learn_apply_quiz(
            "```json\n{\"explain\":[\"App owns the panel state.\"],\"question\":\"Which type owns the plan?\",\"options\":[\"App\",\"Plan\",\"Frame\"],\"answer\":1,\"why\":\"The plan field is a Plan.\"}\n```",
        );
        assert!(app.learn_quiz_active);
        assert_eq!(app.learn_quiz_answer, Some(1));
        assert_eq!(app.learn_explain, vec!["App owns the panel state."]);
        assert!(!app.learn_busy, "the reply ended the request");
        app.learn_quiz_selected = 0;
        app.learn_answer_quiz();
        assert!(app.learn_quiz_answered);
        assert!(!app.learn_quiz_correct);
        app.learn_quiz_selected = 1;
        app.learn_quiz_answered = false;
        app.learn_answer_quiz();
        assert!(app.learn_quiz_correct);
        assert_eq!(app.learn_quiz_why, "The plan field is a Plan.");
    }

    #[test]
    fn a_reply_without_a_quiz_is_reported_not_invented() {
        let mut app = App::for_tests();
        app.learn_apply_quiz("Sure! Ownership means each value has one owner.");
        assert!(!app.learn_quiz_active, "no quiz in it, so no quiz shown");
        assert!(
            app.learn_status
                .starts_with("The model did not answer with a quiz"),
            "{}",
            app.learn_status
        );
        assert!(
            app.learn_status.contains("Ownership means"),
            "the reply verbatim"
        );
    }

    #[test]
    fn lesson_quiz_parsing_needs_a_usable_answer_key() {
        let with_answer = |answer: &str| {
            format!("{{\"explain\":[\"a\"],\"question\":\"q\",\"options\":[\"x\",\"y\"],\"answer\":{answer}}}")
        };
        assert_eq!(parse_lesson_quiz(&with_answer("1")).unwrap().answer, 1);
        assert_eq!(
            parse_lesson_quiz(&with_answer("\"0\"")).unwrap().answer,
            0,
            "a quoted index is still an index"
        );
        assert!(
            parse_lesson_quiz(&with_answer("2")).is_none(),
            "past the end"
        );
        assert!(parse_lesson_quiz(
            "{\"explain\":[],\"question\":\"q\",\"options\":[\"x\",\"y\"],\"answer\":0}"
        )
        .is_none());
        assert!(parse_lesson_quiz(
            "{\"explain\":[\"a\"],\"question\":\"q\",\"options\":[\"x\"],\"answer\":0}"
        )
        .is_none());
        assert!(parse_lesson_quiz("prose, no json").is_none());

        assert_eq!(cap_at_line("aaaa\nbbbb\n", 6), "aaaa");
        assert_eq!(cap_at_line("short", 6), "short");
        assert_eq!(cap_at_line("no-newline-at-all", 10), "no-newline");
    }

    /// The other half of J-06's done-when rule: an un-indexed workspace gets
    /// the reason, and nothing is sent to a provider.
    #[test]
    fn learning_without_an_index_says_why_and_asks_nothing() {
        let (tx, _rx) = mpsc::unbounded_channel::<String>();
        let mut app = App::for_tests();
        let root = temp_dir("learn-noindex");
        app.start_learning(root.clone(), tx.clone());
        assert!(app.learn_active, "the panel still opened");
        assert!(app.learn_lessons.is_empty());
        assert_eq!(
            app.learn_status,
            "No project index — run /init first, then Enter again."
        );
        assert!(!app.learn_busy, "nothing was sent to a provider");
        // And walking an empty queue cannot conjure a lesson either.
        app.learn_step(true, tx);
        assert_eq!(app.learn_current_lesson, 0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The meter token carries the reading and the running byte count. A
    /// malformed one must not be allowed to blank the meter mid-sentence.
    #[test]
    fn a_level_reading_needs_both_its_number_and_its_count() {
        assert_eq!(parse_voice_level("0.4210|3200"), Some((0.421, 3200)));
        assert_eq!(parse_voice_level("0.4"), None);
        assert_eq!(parse_voice_level("loud|3200"), None);
        assert_eq!(parse_voice_level("0.4|many"), None);

        let mut app = App::for_tests();
        app.voice_apply_level("0.4210|3200");
        assert_eq!(app.voice_level, 0.421);
        assert_eq!(app.voice_pcm_bytes, 3200);
        app.voice_apply_level("garbage");
        assert_eq!(app.voice_level, 0.421, "a bad reading changes nothing");
    }

    /// The peak is the loudest chunk of the session, so a bar that has since
    /// fallen is still visible as what the clip reached.
    #[test]
    fn the_voice_peak_holds_the_loudest_chunk() {
        let mut app = App::for_tests();
        app.voice_apply_level("0.6000|3200");
        app.voice_apply_level("0.2000|6400");
        assert_eq!(app.voice_level, 0.2);
        assert_eq!(app.voice_peak, 0.6);
        assert_eq!(app.voice_pcm_bytes, 6400);
    }

    /// A clip report is the honest end of a session with no speech engine: the
    /// panel keeps the file it wrote and says it cannot transcribe. The
    /// transcript list stays empty because nobody transcribed anything.
    #[test]
    fn a_clip_without_a_transcriber_reports_the_file_not_a_sentence() {
        let mut app = App::for_tests();
        app.voice_active = true;
        app.voice_apply_clip("/tmp/.xencode/voice/clip-1.wav|1500");
        assert_eq!(
            app.voice_clip.as_ref().unwrap().display().to_string(),
            "/tmp/.xencode/voice/clip-1.wav"
        );
        assert!(app.voice_note.contains("1.5 s"), "{}", app.voice_note);
        assert!(app.voice_transcript.is_empty());

        app.voice_apply_note(&crate::voice::missing_transcriber_note(
            std::path::Path::new("/tmp/.xencode/voice/clip-1.wav"),
        ));
        assert!(
            app.voice_note.contains("No speech-to-text engine"),
            "{}",
            app.voice_note
        );
        assert!(
            app.voice_transcript.is_empty(),
            "nothing was ever transcribed"
        );

        // A clip report the panel cannot parse is said out loud, not dropped.
        app.voice_apply_clip("/tmp/clip-2.wav|soon");
        assert!(app.voice_note.contains("Malformed"), "{}", app.voice_note);
        assert_eq!(
            app.voice_clip.as_ref().unwrap().display().to_string(),
            "/tmp/.xencode/voice/clip-1.wav"
        );
    }

    #[test]
    fn a_failed_capture_stops_the_meter_and_says_why() {
        let mut app = App::for_tests();
        app.voice_busy = true;
        app.voice_status = "listening".into();
        app.voice_apply_level("0.5000|3200");
        app.voice_apply_error("arecord failed to start: No such device");
        assert!(!app.voice_busy);
        assert_eq!(app.voice_status, "idle");
        assert_eq!(app.voice_level, 0.0);
        assert!(
            app.voice_note.contains("No such device"),
            "{}",
            app.voice_note
        );
    }

    /// Mute has to reach the thread that is reading the microphone, otherwise
    /// `m` is a label and the clip still gets written.
    #[test]
    fn muting_switches_the_flag_the_capture_thread_reads() {
        use std::sync::atomic::Ordering;
        let mut app = App::for_tests();
        assert!(!app.voice_mute_flag.load(Ordering::Relaxed));
        app.set_voice_muted(true);
        assert!(app.voice_muted);
        assert!(app.voice_mute_flag.load(Ordering::Relaxed));
        app.set_voice_muted(false);
        assert!(!app.voice_mute_flag.load(Ordering::Relaxed));
    }

    /// A recorder that is not a recorder — the empty PCM case — produces no
    /// clip at all, so there is nothing for the panel to claim it kept.
    #[test]
    fn a_capture_of_nothing_keeps_no_clip() {
        let dir = temp_dir("voice-empty");
        let pcm_file = dir.join("nothing.pcm");
        std::fs::write(&pcm_file, []).unwrap();
        let cat = crate::voice::which("cat").expect("cat is on PATH for this test");
        let args = vec![pcm_file.into_os_string()];
        let cap = crate::voice::capture(
            &cat,
            &args,
            &std::sync::atomic::AtomicBool::new(false),
            &std::sync::atomic::AtomicBool::new(false),
            |_, _| {},
        );
        assert!(cap.error.is_none(), "{:?}", cap.error);
        assert!(cap.pcm.is_empty());
        assert!(cap.levels.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Every number the profiler panel shows has to be measured or explained.
    /// CPU is a rate so it may legitimately fail to read; what it may not do
    /// is fall back to the scripted table this replaced.
    #[tokio::test]
    async fn profiler_measures_the_process_and_its_metrics() {
        let dir = temp_dir("profiler");
        let row = xencode_context_rs::RequestMetrics::from_timings(
            "BALANCED", 8192, 4000, 400, 120, 31.5, 900.0, 5,
        );
        xencode_context_rs::append_metrics(&dir, &row).unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        super::run_profiler(dir.clone(), tx).await;
        let mut messages = Vec::new();
        while let Ok(m) = rx.try_recv() {
            messages.push(m);
        }
        let _ = std::fs::remove_dir_all(&dir);

        // CPU and memory: a measured row, or an honest note about why not.
        for what in ["cpu", "resident"] {
            let measured = messages.iter().any(|m| m.contains(what));
            let admitted = messages
                .iter()
                .any(|m| m.starts_with("[PROFILER]note:") && m.contains("could not be read"));
            assert!(
                measured || admitted,
                "neither {} nor a note: {:?}",
                what,
                messages
            );
        }
        assert!(messages
            .iter()
            .any(|m| m.starts_with("[PROFILER]gauge:cpu|")));
        // The recorded turn, straight out of metrics.jsonl.
        assert!(
            messages.iter().any(|m| {
                m.starts_with("[PROFILER]row:BALANCED|")
                    && m.contains("90% kv")
                    && m.contains("4000 prompt")
                    && m.contains("32 tok/s")
                    && m.contains("5 files")
            }),
            "no metrics row in {messages:?}"
        );
        assert!(messages.iter().any(|m| m == "[PROFILER]done"));
        for scripted in ["process_data", "render_template", "generate_report"] {
            assert!(
                !messages.iter().any(|m| m.contains(scripted)),
                "profiler still lists {scripted}"
            );
        }
    }

    #[tokio::test]
    async fn profiler_says_so_without_recorded_metrics() {
        let dir = temp_dir("profiler-empty");
        let (tx, mut rx) = mpsc::unbounded_channel();
        super::run_profiler(dir.clone(), tx).await;
        let mut messages = Vec::new();
        while let Ok(m) = rx.try_recv() {
            messages.push(m);
        }
        let _ = std::fs::remove_dir_all(&dir);

        assert!(messages
            .iter()
            .any(|m| m.starts_with("[PROFILER]note:no metrics.jsonl")));
        assert!(!messages
            .iter()
            .any(|m| m.starts_with("[PROFILER]row:metrics|")));
    }

    /// The app-side half of the panel: session numbers exist, and latency
    /// without a completed turn reads as `n/a` rather than 0 ms.
    #[tokio::test]
    async fn profiler_rows_come_from_app_state() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        let lines = app.messages.len();
        app.messages.push(super::UiMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        });
        app.start_profiler(tx);

        assert!(app
            .profiler_rows
            .iter()
            .any(|(s, m, _)| s == "session" && m == "uptime"));
        let counted = app
            .profiler_rows
            .iter()
            .find(|(s, m, _)| s == "session" && m == "chat lines")
            .expect("no chat-lines row");
        assert_eq!(counted.2, format!("{}", lines + 1));
        assert_eq!(
            app.profiler_gauge_latency, None,
            "latency must stay unknown before the first turn"
        );
        assert!(app
            .profiler_notes
            .iter()
            .any(|n| n.contains("no completed turn yet")));
    }

    /// J-03: the model's reply is the only source of a suggestion — this parser
    /// is where the scripted five-command list used to be built. It accepts what
    /// models actually send (fences, prose, a bare object) and a flattering risk
    /// label can only ever be raised, never lowered.
    #[test]
    fn term_suggestions_come_from_the_reply_and_risk_only_escalates() {
        let fenced = parse_term_suggestions(
            "Sure:\n```json\n[{\"command\":\"du -sh *\",\"risk\":\"safe\",\"why\":\"sizes\"}]\n```",
        );
        assert_eq!(
            fenced,
            vec![(
                "du -sh *".to_string(),
                "safe".to_string(),
                "sizes".to_string()
            )]
        );

        let escalated =
            parse_term_suggestions("[{\"command\":\"rm -rf node_modules\",\"risk\":\"safe\"}]");
        assert_eq!(
            escalated[0].1, "destructive",
            "a command on the dangerous list is never shown as safe"
        );

        let kept = parse_term_suggestions(
            "[{\"command\":\"git push --force\",\"explanation\":\"own warning, other key\"}]",
        );
        assert_eq!(kept[0].1, "destructive");
        assert_eq!(kept[0].2, "own warning, other key");

        assert!(parse_term_suggestions("I would not run anything for that.").is_empty());
        assert!(parse_term_suggestions("[{\"why\":\"no command in this object\"}]").is_empty());

        let items: Vec<String> = (0..12)
            .map(|i| format!("{{\"command\":\"echo {i}\",\"risk\":\"safe\"}}"))
            .collect();
        assert_eq!(
            parse_term_suggestions(&format!("[{}]", items.join(","))).len(),
            super::TERM_SUGGESTION_CAP,
            "a wall of suggestions is not a list anyone can read"
        );
    }

    /// The panel starts empty (nothing canned is on screen before the model has
    /// answered) and its filter can only ever narrow to rows that exist.
    #[test]
    fn terminal_panel_opens_asked_for_and_filters_by_risk() {
        let app = App::for_tests();
        assert!(
            app.term_asst_suggestions.is_empty(),
            "the panel no longer ships a scripted list"
        );
        assert!(app.term_asst_typing, "it opens ready to be typed into");
        assert!(app.term_visible_rows().is_empty());

        let mut app = App::for_tests();
        app.term_asst_suggestions = vec![
            ("df -h".to_string(), "safe".to_string(), String::new()),
            (
                "rm -rf ./build".to_string(),
                "destructive".to_string(),
                String::new(),
            ),
        ];
        app.term_risk_filter = "destructive".to_string();
        assert_eq!(app.term_visible_rows(), vec![1]);
        app.term_risk_filter = "safe".to_string();
        assert_eq!(app.term_visible_rows(), vec![0]);
        app.term_risk_filter = "All".to_string();
        assert_eq!(app.term_visible_rows(), vec![0, 1]);
    }

    /// An empty question must not spend a provider call — the panel asks when
    /// the user asks, and never on its own.
    #[test]
    fn asking_nothing_sends_nothing() {
        let mut app = App::for_tests();
        let (tx, mut rx) = mpsc::unbounded_channel();
        app.ask_terminal(tx);
        assert!(!app.term_asst_busy);
        assert!(app.term_asst_output.contains("Nothing asked"));
        assert!(
            rx.try_recv().is_err(),
            "a question that was never asked must not reach a provider"
        );
    }

    /// The panel's only route to a shell is the agent's own gate: an unapproved
    /// command comes back as a denial and never executes (J-03).
    #[tokio::test]
    async fn terminal_panel_runs_commands_through_the_approval_gate() {
        let mut app = App::for_tests();
        app.config.agent_approval = "ask".to_string();
        let marker = std::env::temp_dir().join(format!(
            "xencode-term-gate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        app.term_asst_suggestions = vec![(
            format!("touch {}", marker.display()),
            "safe".to_string(),
            String::new(),
        )];

        let (tx, mut done_rx) = mpsc::unbounded_channel();
        app.run_terminal_suggestion(tx);

        let (request, responder) = app
            .approval_rx
            .as_mut()
            .expect("no approval channel")
            .recv()
            .await
            .expect("the panel ran a command without asking");
        assert_eq!(request.tool, "run_command");
        let _ = responder.send(crate::agent_tools::ApprovalAnswer::Denied);

        let token = done_rx.recv().await.expect("no [TERM] result");
        let json = token
            .strip_prefix("[TERM]ran:")
            .unwrap_or_default()
            .to_string();
        let body: serde_json::Value = serde_json::from_str(&json).expect("result is not JSON");
        assert!(
            body["result"]
                .as_str()
                .unwrap_or_default()
                .starts_with("error: the user denied"),
            "a denial has to read as a denial: {body}"
        );
        assert!(!marker.exists(), "a denied command must never run");
    }

    /// Same gate, no listener: with nothing to answer the prompt, the strictest
    /// possible answer is the one the panel gets.
    #[tokio::test]
    async fn a_vanished_prompter_denies_instead_of_running() {
        let mut app = App::for_tests();
        app.config.agent_approval = "ask".to_string();
        app.term_asst_suggestions = vec![(
            "touch /tmp/xencode-panel-must-not-run".to_string(),
            "safe".to_string(),
            String::new(),
        )];
        app.approval_rx = None;

        let (tx, mut done_rx) = mpsc::unbounded_channel();
        app.run_terminal_suggestion(tx);
        let token = done_rx.recv().await.expect("no [TERM] result");
        let body: serde_json::Value =
            serde_json::from_str(token.trim_start_matches("[TERM]ran:")).unwrap();
        assert!(body["result"]
            .as_str()
            .unwrap_or_default()
            .starts_with("error: the user denied"));
        assert!(!std::path::Path::new("/tmp/xencode-panel-must-not-run").exists());
    }

    fn git(root: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
