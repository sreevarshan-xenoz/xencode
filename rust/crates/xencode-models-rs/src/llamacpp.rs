use std::fmt;
use std::process::Stdio;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::health::{HealthStatus, HealthTracker, ModelHealth};

/// Information about a model hosted in llama.cpp (`llama-server`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaCppModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    #[serde(default)]
    data: Vec<OpenAIModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelEntry {
    id: String,
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    owned_by: Option<String>,
}

/// Completion timing and token usage reported by a llama.cpp server.
///
/// The three token counts come from the server's own `usage` object on a
/// completion response. Measured on `llama-server` b10809 hosting a 0.6B Q4_K_M
/// GGUF, with a 23,003-character prompt made of a 4,742-character stable head,
/// 18,197 characters of retrieved file bodies and a 60-character question:
///
/// | what was asked | what came back |
/// | --- | --- |
/// | the prompt as one chat request | `prompt_tokens: 5766` |
/// | the same text, no conversation sent yet | `cached_tokens: 1` |
/// | that request sent again, unchanged | `cached_tokens: 5765` |
/// | the same text through `/tokenize` | 5753 tokens |
///
/// `prompt_tokens` is therefore the whole prompt including the chat template's
/// framing, and the difference from a plain `/tokenize` of the same text is 13
/// tokens for a two-message request — the template, not the content.
/// `cached_tokens` is a reading of the server's own memory rather than arithmetic,
/// which is why it is kept: it is the only number here that says whether the
/// byte-stable prefix did its job.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppTimings {
    /// Number of tokens generated.
    pub tokens_generated: u64,
    /// Prompt tokens the server had to evaluate this time, which is its prompt
    /// total minus whatever prefix it reused.
    pub tokens_evaluated: u64,
    /// Whole prompt size as the server counted it, chat template included.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Prompt tokens reused from the server's own memory rather than evaluated.
    #[serde(default)]
    pub cached_tokens: u64,
    /// Generation throughput in tokens per second (tok/s).
    pub predicted_per_second: f64,
    /// Prompt processing throughput in tokens per second. A completion response
    /// says how many prompt tokens there were, not how long evaluating them took,
    /// so this stays unmeasured unless a server says otherwise.
    pub prompt_per_second: f64,
    /// Total wall clock time of the generation in seconds.
    pub total_seconds: f64,
}

impl LlamaCppTimings {
    /// Build the record from what a completion response's `usage` said.
    ///
    /// `prompt_tokens` and `cached_tokens` arrive as the server's counts;
    /// `tokens_evaluated` is derived from them rather than reported, and
    /// saturating so a server that claims more reuse than prompt cannot produce
    /// a negative evaluation.
    pub fn from_usage(
        generated: u64,
        prompt_tokens: u64,
        cached_tokens: u64,
        elapsed: f64,
    ) -> Self {
        Self {
            tokens_generated: generated,
            tokens_evaluated: prompt_tokens.saturating_sub(cached_tokens),
            prompt_tokens,
            cached_tokens,
            predicted_per_second: if elapsed > 0.0 {
                generated as f64 / elapsed
            } else {
                0.0
            },
            prompt_per_second: 0.0,
            total_seconds: elapsed,
        }
    }
}

#[derive(Debug, Deserialize)]
struct LlamaCppProps {
    #[serde(default)]
    default_generation_settings: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    total_slots: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct LlamaCppLoadResponse {
    #[serde(default)]
    error: Option<String>,
}

/// Pick the context window out of a `/props` response.
///
/// Measured on `llama-server` b10809: the running window is
/// `default_generation_settings.n_ctx`, and `/props` has **no** top-level
/// `n_ctx` — a build that starts the server with `-c 8192` reports 8192 there
/// and nothing elsewhere. The top-level key is read as a second candidate
/// because releases before that one carried it there; that shape is unverified
/// on this machine, so the nested key stays first and the whole response is
/// checked before either is believed.
///
/// Returns `None` when no candidate is a positive number that fits a `u32`.
pub fn context_window_from_props(props: &serde_json::Value) -> Option<u32> {
    if let Some(nested) = props
        .get("default_generation_settings")
        .and_then(|dgs| dgs.get("n_ctx"))
        .and_then(|v| v.as_u64())
    {
        if let Some(window) = as_window(nested) {
            return Some(window);
        }
    }
    props
        .get("n_ctx")
        .and_then(|v| v.as_u64())
        .and_then(as_window)
}

fn as_window(tokens: u64) -> Option<u32> {
    if tokens == 0 {
        return None;
    }
    u32::try_from(tokens).ok()
}

/// What a running server says about itself, read out of one `/props` response.
///
/// A field is `None` when the server answered without reporting it, which is a
/// different answer from `Some` and stays that way in the caller's wording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerReport {
    /// Tokens of context per slot — see [`context_window_from_props`].
    ///
    /// Per slot, not total: measured on b10809, a server started with
    /// `--ctx-size 2048 --parallel 2` reported 1024 here.
    pub context_tokens: Option<u32>,
    /// How many conversations the server serves at once (`total_slots`).
    pub slots: Option<u32>,
}

/// Pick both reported values out of a `/props` response.
///
/// Measured on b10809: `default_generation_settings.n_ctx` and a top-level
/// `total_slots` are both present; the launch's `--cache-type-k`,
/// `--cache-type-v` and `--batch-size` appear nowhere in the response, so a
/// preset that includes them cannot be checked this way.
pub fn report_from_props(props: &serde_json::Value) -> ServerReport {
    ServerReport {
        context_tokens: context_window_from_props(props),
        slots: props
            .get("total_slots")
            .and_then(|v| v.as_u64())
            .and_then(|s| u32::try_from(s).ok())
            .filter(|s| *s > 0),
    }
}

/// Pick the token count out of a `/tokenize` response, for the `text` that was
/// sent to get it.
///
/// Measured on `llama-server` b10809: the response is `{"tokens":[…]}`, one
/// integer per token, so the answer is the length of that array. `None` covers
/// both a body without such an array and — the case that makes `text` part of
/// this function — an empty array given back for text that was not empty. That
/// is what b10809 does to a request using a field name it does not read, and it
/// answers HTTP 200 while doing it, so believing zero here would report a prompt
/// as empty instead of reporting that nobody counted it.
pub fn counted_tokens_from_response(body: &serde_json::Value, text: &str) -> Option<u64> {
    let tokens = body.get("tokens").and_then(|v| v.as_array())?;
    if tokens.is_empty() && !text.is_empty() {
        return None;
    }
    Some(tokens.len() as u64)
}

/// Say what a server xencode started reports about itself, next to what it was
/// started with.
///
/// `/props` is the only witness, and it carries two of the preset's values —
/// the window and the slot count. The cache quantization and batch size appear
/// nowhere in the response, so this says nothing about them either way.
pub fn settings_check_line(label: &str, asked: ServerReport, got: ServerReport) -> String {
    if got.context_tokens.is_none() && got.slots.is_none() {
        return format!("{label}: the server reported no settings, so nothing is verified");
    }
    let summary = |report: ServerReport| match (report.context_tokens, report.slots) {
        (Some(ctx), Some(slots)) => format!("{ctx} tokens of context in {slots} slot(s)"),
        (Some(ctx), None) => format!("{ctx} tokens of context"),
        (None, Some(slots)) => format!("{slots} slot(s)"),
        (None, None) => String::new(),
    };
    let got_summary = summary(got);
    let asked_summary = summary(asked);
    if asked_summary.is_empty() {
        return format!("{label}: {got_summary}");
    }
    // Only a value both sides have can disagree; a server that reports half of
    // what was asked has not contradicted the other half.
    let contradicted = asked
        .context_tokens
        .zip(got.context_tokens)
        .is_some_and(|(want, reported)| want != reported)
        || asked
            .slots
            .zip(got.slots)
            .is_some_and(|(want, reported)| want != reported);
    if !contradicted {
        return format!("{label}: {got_summary}, as asked");
    }
    // A flag written twice is decided by the last value, so a difference usually
    // means `llama_cpp_args` had the last word rather than a preset failing to
    // apply.
    format!(
        "{label}: {got_summary}, not the {asked_summary} asked for — later flags win, so check llama_cpp_args"
    )
}

/// How a server process that xencode started came to an end.
///
/// The two shapes matter separately. A server that exits on its own knows why
/// and said so; a server the kernel's out-of-memory killer takes is stopped
/// dead with no line of its own, which is why the signal is reported as loudly
/// as the message. A shell adds the same fact twice over: `kill -9` is signal
/// 9, and the `137` a script sees is that number plus 128.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerExit {
    /// The exit code, when the process got as far as setting one.
    pub code: Option<i32>,
    /// The signal that ended it, when one did.
    pub signal: Option<i32>,
}

impl ServerExit {
    fn new(status: &std::process::ExitStatus) -> Self {
        ServerExit {
            code: status.code(),
            signal: exit_signal(status),
        }
    }

    /// One line saying how it died, in the words a user can act on.
    pub fn describe(&self) -> String {
        match (self.signal, self.code) {
            (Some(9), _) => {
                "killed by signal 9 — the kernel's out-of-memory killer sends this one".to_string()
            }
            (Some(signal), _) => format!("killed by signal {signal}"),
            (None, Some(code)) => format!("exited with code {code}"),
            (None, None) => "ended without reporting anything".to_string(),
        }
    }

    /// Whether this death looks like a memory problem rather than a bad command
    /// line. The signal alone counts, because the kernel's killer gives the
    /// process no chance to explain itself; beyond that it is the server's own
    /// words.
    ///
    /// The comparison drops spaces, underscores and punctuation on purpose: the
    /// same fact reaches the terminal as `out of memory`, `std::bad_alloc` and
    /// — what this build of `llama-server` printed when a device ran out,
    /// measured here — `ggml_vulkan: vk::Device::allocateMemory:
    /// ErrorOutOfDeviceMemory`. Each of those compacts to one of the words
    /// below; a plain substring match against any of them misses two of three.
    pub fn looks_like_out_of_memory(&self, tail: &[String]) -> bool {
        if self.signal == Some(9) {
            return true;
        }
        const MEMORY_WORDS: [&str; 5] = [
            "outofmemory",
            "outofdevicememory",
            "cannotallocate",
            "failedtoallocate",
            "badalloc",
        ];
        tail.iter().any(|line| {
            let compacted: String = line
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect();
            MEMORY_WORDS.iter().any(|word| compacted.contains(word))
        })
    }
}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// How many of a server's own error lines are kept for reporting.
const LOG_TAIL_LINES: usize = 40;

/// The last few lines a server wrote to its own error output.
///
/// A reader thread fills this while the process runs and stops at the end of
/// the pipe, so what killed the server is still there to be quoted after it is
/// gone. The list is capped: this exists to explain a failure, not to hold a
/// log, and an unbounded one is a slow leak in a program that starts a server
/// per session.
#[derive(Debug, Clone, Default)]
pub struct ServerLog {
    lines: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
}

impl ServerLog {
    fn push(&self, line: String) {
        if let Ok(mut lines) = self.lines.lock() {
            if lines.len() == LOG_TAIL_LINES {
                lines.pop_front();
            }
            lines.push_back(line);
        }
    }

    /// What the server said last, oldest first.
    pub fn tail(&self) -> Vec<String> {
        self.lines
            .lock()
            .map(|lines| lines.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// A running `llama-server` process that xencode spawned (auto-start support).
#[derive(Debug)]
pub struct LlamaServerProcess {
    child: std::process::Child,
    pub base_url: String,
    log: ServerLog,
    /// The thread copying the server's error output, kept so it can be waited
    /// for. Reading the tail before it finishes is how a report ends up quoting
    /// half a sentence — or none of the sentence that explained the failure.
    reader: Option<std::thread::JoinHandle<()>>,
}

impl LlamaServerProcess {
    /// The OS process id of the spawned server (if available).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Check whether the spawned server is still running.
    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().map(|s| s.is_none()).unwrap_or(false)
    }

    /// How the server ended, if it has. `None` while it is still running.
    ///
    /// This is the check a launch loop is supposed to make before it keeps
    /// waiting: a process that died at the third second does not become ready
    /// in the hundred and twentieth, and waiting the difference out reports a
    /// timeout as if that were what happened.
    pub fn exit_status(&mut self) -> Option<ServerExit> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(ServerExit::new(&status)),
            Ok(None) => None,
            // Cannot ask; treat as still running and let the caller's own
            // deadline decide, rather than claiming a death that was not seen.
            Err(_) => None,
        }
    }

    /// What the server said on its own error output. Waiting for the reader
    /// first, so the answer is the whole tail rather than the part of it that
    /// had arrived by the time the process disappeared.
    pub fn log(&mut self) -> Vec<String> {
        self.drain_log();
        self.log.tail()
    }

    fn drain_log(&mut self) {
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }

    /// Terminate the spawned server process.
    pub fn stop(&mut self) -> Result<(), LlamaCppError> {
        self.child
            .kill()
            .map_err(|e| LlamaCppError::Api(format!("failed to stop llama-server: {e}")))?;
        // Reap it: a child left dead but unwaited-for stays in the process
        // table, and `wait` is also what closes the error pipe the reader
        // thread is holding open.
        self.child
            .wait()
            .map_err(|e| LlamaCppError::Api(format!("failed to stop llama-server: {e}")))?;
        self.drain_log();
        Ok(())
    }

    /// Start waiting where a launch left off: ask the server if its model is
    /// in, and stop asking the moment it is clear that it never will be.
    ///
    /// The liveness check inside the loop is the point. A launch that only
    /// counts attempts cannot tell a server that is still loading a model from
    /// one that died three seconds in, so it waits out the whole deadline and
    /// then reports the one thing that was never true — that it timed out.
    /// Readiness is asked for strictly for the same reason: a server that
    /// answers `Loading model` and then dies during the cache allocation is
    /// exactly the failure this loop exists to catch.
    pub async fn wait_until_ready(
        &mut self,
        tries: u32,
        gap: std::time::Duration,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> ServerStart {
        let client = LlamaCppClient::new(&self.base_url, 3);
        for _ in 0..tries {
            if cancelled() {
                let _ = self.stop();
                return ServerStart::Cancelled;
            }
            if client.model_ready().await {
                return ServerStart::Ready;
            }
            if let Some(exit) = self.exit_status() {
                self.drain_log();
                return ServerStart::Died {
                    exit,
                    tail: self.log.tail(),
                };
            }
            tokio::time::sleep(gap).await;
        }
        match self.exit_status() {
            Some(exit) => {
                self.drain_log();
                ServerStart::Died {
                    exit,
                    tail: self.log.tail(),
                }
            }
            None => ServerStart::NotReady,
        }
    }
}

/// What waiting for a freshly started server turned out to mean.
#[derive(Debug, Clone)]
pub enum ServerStart {
    /// It is answering.
    Ready,
    /// The process went away. `tail` is what it said last, which is the only
    /// account of why that exists.
    Died { exit: ServerExit, tail: Vec<String> },
    /// Still alive, still not answering.
    NotReady,
    /// xencode is going away, and the server was stopped on the way out.
    Cancelled,
}

/// What starting a server and waiting for it turned out to mean.
#[derive(Debug)]
pub enum LaunchOutcome {
    /// It is answering. `notes` is what the machine said along the way — a
    /// launch that had to be made smaller to work is a success worth reading,
    /// because the next person to change a flag needs to know it was already at
    /// the edge.
    Started {
        server: LlamaServerProcess,
        notes: Vec<String>,
    },
    /// There is no server. Every line here says something true about why,
    /// including the lines the server printed itself.
    Failed { lines: Vec<String> },
    /// xencode is going away; a server that had started was stopped first.
    Cancelled,
}

/// The window a set of launch arguments actually asks for: the last
/// `--ctx-size`/`-c` value in the list, which is the one `llama-server` runs
/// with. `None` when no window is named at all, which means the server's own
/// default and nothing here can halve a number that was never written.
pub fn ctx_size_in(args: &[String]) -> Option<u64> {
    let mut found = None;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        let value = if arg == "--ctx-size" || arg == "-c" {
            index += 1;
            args.get(index).map(|s| s.as_str())
        } else {
            arg.strip_prefix("--ctx-size=")
                .or_else(|| arg.strip_prefix("-c="))
        };
        if let Some(value) = value {
            if let Ok(tokens) = value.trim().parse::<u64>() {
                found = Some(tokens);
            }
        }
        index += 1;
    }
    found
}

/// How long a launch is willing to wait for a server to answer: a number of
/// attempts and the pause between them. Loading a model is the slow part this
/// exists for, and it is slow by the size of the file rather than by anything
/// xencode controls.
#[derive(Debug, Clone, Copy)]
pub struct Patience {
    pub tries: u32,
    pub gap: std::time::Duration,
}

/// Start a server, wait for it to answer, and — if the machine said it ran out
/// of memory — try once more at a shorter window before giving up.
///
/// This is what both launch sites in xencode were missing. Each of them counted
/// attempts without ever looking at the process, so a server that died during
/// model loading was reported two minutes later as one that "did not become
/// ready in time" — a timeout, which is not what happened, and which points
/// whoever reads it at the wrong thing. Waiting on the process instead means the
/// first answer is either "it's up" or "it's gone, and here is what it said".
///
/// `step_down` is the budget layer's arithmetic passed in rather than
/// duplicated: it is given the window in force and returns a smaller one, or
/// `None` when there is nothing left to give. Only one retry is made, because
/// the failure this is aimed at is the machine's size, and a second attempt at
/// half of a half asks the same question.
pub async fn launch_and_wait(
    executable: &str,
    model_path: &str,
    port: u16,
    args: &[String],
    patience: Patience,
    cancelled: &(dyn Fn() -> bool + Sync),
    step_down: &(dyn Fn(u64) -> Option<u64> + Sync),
) -> LaunchOutcome {
    let Patience { tries, gap } = patience;
    let mut args = args.to_vec();
    let mut notes: Vec<String> = Vec::new();
    let mut retried = false;

    loop {
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let mut server = match start_llama_server(executable, model_path, port, &refs) {
            Ok(server) => server,
            Err(e) => {
                return LaunchOutcome::Failed {
                    lines: vec![format!("could not start llama-server: {e}")],
                }
            }
        };
        match server.wait_until_ready(tries, gap, cancelled).await {
            ServerStart::Ready => {
                return LaunchOutcome::Started { server, notes };
            }
            ServerStart::Cancelled => return LaunchOutcome::Cancelled,
            ServerStart::NotReady => {
                // Still running, still silent. The log is the only place a
                // reason could be, and it is quoted rather than summarised.
                let tail = server.log();
                let _ = server.stop();
                let mut lines = vec![format!(
                    "llama-server is still running after {} attempts and has not answered on \
                     port {port}",
                    pretty_attempts(tries, gap)
                )];
                lines.extend(quote_tail(&tail));
                return LaunchOutcome::Failed { lines };
            }
            ServerStart::Died { exit, tail } => {
                if !exit.looks_like_out_of_memory(&tail) {
                    let mut lines = vec![format!(
                        "llama-server stopped before it answered: {}",
                        exit.describe()
                    )];
                    lines.extend(quote_tail(&tail));
                    return LaunchOutcome::Failed { lines };
                }
                // Out of memory, which is the one failure here that a smaller
                // request can answer. What was asked is whatever the arguments
                // in force name, including the value this function appended on
                // its own retry.
                let in_force = ctx_size_in(&args);
                let smaller = in_force.and_then(step_down);
                if retried {
                    let asked = in_force
                        .map(|tokens| tokens.to_string())
                        .unwrap_or_else(|| "the window asked for".to_string());
                    let mut lines = vec![format!(
                        "llama-server ran out of memory at {asked} tokens as well, so this \
                         machine cannot serve {model_path} with a window worth having"
                    )];
                    lines.extend(std::mem::take(&mut notes));
                    lines.extend(quote_tail(&tail));
                    lines.extend(escalation());
                    return LaunchOutcome::Failed { lines };
                }
                match (in_force, smaller) {
                    (Some(first), Some(second)) => {
                        notes.push(format!(
                            "llama-server ran out of memory at {first} tokens; starting again at \
                             {second}."
                        ));
                        retried = true;
                        args.push("--ctx-size".to_string());
                        args.push(second.to_string());
                    }
                    (Some(first), None) => {
                        let mut lines = vec![format!(
                            "llama-server ran out of memory at {first} tokens and there is no \
                             shorter window to try"
                        )];
                        lines.extend(quote_tail(&tail));
                        lines.extend(escalation());
                        return LaunchOutcome::Failed { lines };
                    }
                    (None, _) => {
                        let mut lines = vec![
                            "llama-server ran out of memory, and no context size was asked for, \
                             so there is no window here to make smaller"
                                .to_string(),
                        ];
                        lines.extend(quote_tail(&tail));
                        lines.extend(escalation());
                        return LaunchOutcome::Failed { lines };
                    }
                }
            }
        }
    }
}

/// How a launch attempt's patience would be described in a sentence.
fn pretty_attempts(tries: u32, gap: std::time::Duration) -> String {
    let total = gap * tries;
    format!("{tries} attempts over {:.0} seconds", total.as_secs_f64())
}

/// The last lines the server printed, kept verbatim and attributed. Paraphrasing
/// these is how a report ends up claiming something the program never said.
fn quote_tail(tail: &[String]) -> Vec<String> {
    let lines: Vec<&str> = tail
        .iter()
        .map(|s| s.as_str())
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(6)
        .collect();
    if lines.is_empty() {
        return vec!["the server printed nothing before it stopped".to_string()];
    }
    let mut out = vec![format!(
        "the server's own last {} line{}:",
        lines.len(),
        if lines.len() == 1 { "" } else { "s" }
    )];
    out.extend(lines.into_iter().rev().map(|line| format!("  {line}")));
    out
}

/// What to do when this machine is simply too small. Only commands that exist
/// are named, and the one thing that does not exist yet is said plainly.
fn escalation() -> Vec<String> {
    vec![
        "`xencode hw probe` prints what this machine can hold and the flags to start a server \
         with; `xencode colab up` runs the model on a machine that can hold it, and \
         `xencode config set remote_base_url <url>` points xencode at a server already running \
         elsewhere."
            .to_string(),
        "A smaller quantization of the same model is the other way onto this card; xencode has no \
         model downloader yet, so that file has to be fetched outside it."
            .to_string(),
    ]
}

/// Errors from llama.cpp operations.
#[derive(Debug)]
pub enum LlamaCppError {
    NotRunning(String),
    ModelNotFound(String),
    Timeout(String),
    Api(String),
    Parse(String),
}

impl fmt::Display for LlamaCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaCppError::NotRunning(msg) => write!(f, "llama.cpp server not running: {msg}"),
            LlamaCppError::ModelNotFound(name) => write!(f, "model not found in llama.cpp: {name}"),
            LlamaCppError::Timeout(msg) => write!(f, "request to llama.cpp timed out: {msg}"),
            LlamaCppError::Api(msg) => write!(f, "llama.cpp API error: {msg}"),
            LlamaCppError::Parse(msg) => write!(f, "llama.cpp parse error: {msg}"),
        }
    }
}

impl std::error::Error for LlamaCppError {}

/// Advanced sampling and constraint options supported natively by llama.cpp.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppOptions {
    /// GBNF grammar string for strictly constrained decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// JSON schema for structured JSON output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    /// Min-P sampling threshold (e.g. 0.05).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// Top-K sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Mirostat sampling mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Sampler seed. llama.cpp draws tokens from this, so a run is only
    /// repeatable if the seed goes over the wire with the temperature — leave it
    /// out and the server picks one per request, which is what it does today.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Max tokens to predict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// Client for interacting with a llama.cpp HTTP server (`llama-server`).
pub struct LlamaCppClient {
    base_url: String,
    pub health_tracker: HealthTracker,
    client: reqwest::Client,
}

impl LlamaCppClient {
    /// Create a new client pointing at the given llama.cpp server.
    pub fn new(base_url: &str, timeout_seconds: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .unwrap_or_default();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            health_tracker: HealthTracker::new(),
            client,
        }
    }

    /// Create a client with default settings (http://localhost:8080, 30s timeout).
    pub fn default_client() -> Self {
        Self::new("http://localhost:8080", 30)
    }

    /// Check if llama.cpp server is reachable and responsive, returning roundtrip latency in seconds.
    /// Whether the server has its model in and will answer a request — as
    /// opposed to having merely opened its socket.
    ///
    /// Measured on b10809 at two and three seconds into a launch: while the
    /// weights and the key-value cache are being put in place `/health` answers
    /// 503 `{"error":{"message":"Loading model", …}}`, and when it is done it
    /// answers 200 `{"status":"ok"}`. [`ping`](Self::ping) counts both, because
    /// for a status line "something is there" is the honest answer — but a
    /// launch that treats the first as readiness calls a server running seconds
    /// before it fails to allocate the cache, and then never notices it died.
    pub async fn model_ready(&self) -> bool {
        let health_url = format!("{}/health", self.base_url);
        match self.client.get(&health_url).send().await {
            Ok(r) if r.status().as_u16() == 503 => false,
            Ok(r) if r.status().is_success() => true,
            // A server with no `/health` to ask is as far as this can see, so
            // fall back on the one thing only a loaded model can answer.
            _ => matches!(self.context_window().await, Ok(Some(_))),
        }
    }

    pub async fn ping(&self) -> Result<f64, LlamaCppError> {
        let start = Instant::now();

        // Try /health first
        let health_url = format!("{}/health", self.base_url);
        let resp = self.client.get(&health_url).send().await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(start.elapsed().as_secs_f64()),
            Ok(r) if r.status().as_u16() == 503 => {
                // 503 in llama.cpp indicates server is up but model is currently loading
                Ok(start.elapsed().as_secs_f64())
            }
            _ => {
                // Fallback to /props or /v1/models
                let props_url = format!("{}/props", self.base_url);
                let resp2 = self.client.get(&props_url).send().await.map_err(|e| {
                    if e.is_connect() {
                        LlamaCppError::NotRunning(e.to_string())
                    } else if e.is_timeout() {
                        LlamaCppError::Timeout(e.to_string())
                    } else {
                        LlamaCppError::Api(e.to_string())
                    }
                })?;

                if resp2.status().is_success() {
                    Ok(start.elapsed().as_secs_f64())
                } else {
                    Err(LlamaCppError::Api(format!("HTTP {}", resp2.status())))
                }
            }
        }
    }

    /// List models available on the llama.cpp server.
    pub async fn list_models(&self) -> Result<Vec<LlamaCppModelInfo>, LlamaCppError> {
        let url = format!("{}/v1/models", self.base_url);

        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_connect() {
                LlamaCppError::NotRunning(e.to_string())
            } else if e.is_timeout() {
                LlamaCppError::Timeout(e.to_string())
            } else {
                LlamaCppError::Api(e.to_string())
            }
        })?;

        if response.status().is_success() {
            if let Ok(models_resp) = response.json::<OpenAIModelsResponse>().await {
                if !models_resp.data.is_empty() {
                    return Ok(models_resp
                        .data
                        .into_iter()
                        .map(|m| LlamaCppModelInfo {
                            id: m.id,
                            object: m.object,
                            owned_by: m.owned_by,
                        })
                        .collect());
                }
            }
        }

        // Fallback to /props to check loaded model path
        let props_url = format!("{}/props", self.base_url);
        if let Ok(resp) = self.client.get(&props_url).send().await {
            if resp.status().is_success() {
                if let Ok(props) = resp.json::<LlamaCppProps>().await {
                    if let Some(settings) = props.default_generation_settings {
                        if let Some(model_val) = settings.get("model").and_then(|v| v.as_str()) {
                            let model_name = model_val
                                .replace('\\', "/")
                                .split('/')
                                .next_back()
                                .unwrap_or(model_val)
                                .to_string();
                            return Ok(vec![LlamaCppModelInfo {
                                id: model_name,
                                object: Some("model".to_string()),
                                owned_by: Some("llama.cpp".to_string()),
                            }]);
                        }
                    }
                }
            }
        }

        // If server is responsive, return generic model entry
        if self.ping().await.is_ok() {
            return Ok(vec![LlamaCppModelInfo {
                id: "llamacpp-default".to_string(),
                object: Some("model".to_string()),
                owned_by: Some("llama.cpp".to_string()),
            }]);
        }

        Ok(Vec::new())
    }

    /// Load a GGUF model into the server via POST /v1/models/load.
    ///
    /// `path` is the model identifier as reported by the server (typically a
    /// `--model` or `--alias` value, or a model registry name).
    pub async fn load_model(&self, path: &str) -> Result<(), LlamaCppError> {
        let url = format!("{}/v1/models/load", self.base_url);
        let payload = serde_json::json!({ "model": path });

        let mut resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    LlamaCppError::NotRunning(e.to_string())
                } else if e.is_timeout() {
                    LlamaCppError::Timeout(e.to_string())
                } else {
                    LlamaCppError::Api(e.to_string())
                }
            })?;

        if resp.status().is_success() {
            return Ok(());
        }

        // llama.cpp returns HTTP 503 while a model is loading/swapping; poll until it settles.
        let mut attempts = 0;
        while resp.status().as_u16() == 503 && attempts < 40 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            resp = self
                .client
                .post(&url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| LlamaCppError::Api(e.to_string()))?;
            attempts += 1;
        }

        if resp.status().is_success() {
            return Ok(());
        }

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if let Ok(parsed) = serde_json::from_str::<LlamaCppLoadResponse>(&body) {
            if let Some(ref err) = parsed.error {
                return Err(LlamaCppError::Api(err.clone()));
            }
        }
        Err(LlamaCppError::Api(format!("HTTP {status} - {body}")))
    }

    /// Unload the currently loaded model via POST /v1/models/unload.
    pub async fn unload_models(&self) -> Result<(), LlamaCppError> {
        let url = format!("{}/v1/models/unload", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    LlamaCppError::NotRunning(e.to_string())
                } else if e.is_timeout() {
                    LlamaCppError::Timeout(e.to_string())
                } else {
                    LlamaCppError::Api(e.to_string())
                }
            })?;

        if resp.status().is_success() {
            return Ok(());
        }

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(LlamaCppError::Api(format!(
            "unload failed HTTP {status} - {body}"
        )))
    }

    /// Inline request to swap the loaded model. Convenience wrapper around
    /// [`Self::load_model`].
    pub async fn switch_model(&self, path: &str) -> Result<(), LlamaCppError> {
        self.load_model(path).await
    }

    /// The context window the server is actually running with.
    ///
    /// Reads `/props`; see [`context_window_from_props`] for which keys are
    /// consulted. `Ok(None)` means the server answered without saying — an old
    /// build, or one started with `--props` disabled — which is a different
    /// answer from "no server", and the caller treats it as "keep guessing".
    pub async fn context_window(&self) -> Result<Option<u32>, LlamaCppError> {
        Ok(self.report().await?.context_tokens)
    }

    /// Everything this client knows how to check about a running server, from
    /// one `/props` request.
    pub async fn report(&self) -> Result<ServerReport, LlamaCppError> {
        let url = format!("{}/props", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(ServerReport::default());
        }
        let props = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        Ok(report_from_props(&props))
    }

    /// Ask until the server says something about itself, giving up after `tries`
    /// attempts `gap` apart and returning whatever it said by then.
    ///
    /// This exists because a server that has accepted a connection is not yet a
    /// server that has loaded a model: `llama-server` answers `/health` with 503
    /// while the weights come in, and `/props` with it — measured starting a
    /// 1.5B GGUF, where the settings were unreadable for the first seconds and
    /// reported `4096 tokens in 1 slot` afterwards. A check taken at that moment
    /// has nothing to say, and a client that reports "unverified" for a server
    /// that is merely still loading teaches the user to ignore it.
    pub async fn report_when_ready(&self, tries: u32, gap: std::time::Duration) -> ServerReport {
        for attempt in 0..tries {
            if let Ok(report) = self.report().await {
                if report.context_tokens.is_some() || report.slots.is_some() {
                    return report;
                }
            }
            if attempt + 1 < tries {
                tokio::time::sleep(gap).await;
            }
        }
        ServerReport::default()
    }

    /// How many tokens the server's own vocabulary says `text` is, asked of its
    /// `/tokenize` endpoint. `Ok(None)` means nobody counted it — see
    /// [`counted_tokens_from_response`] for when an answer is not believed.
    ///
    /// Special tokens are counted as the literal text they are written as:
    /// `parse_special` and `add_bos` are accepted and ignored on b10809, so a
    /// marker like `<|end|>` costs five tokens here instead of one. That only
    /// ever pushes the count up, which is the safe direction for a budget.
    pub async fn count_tokens(&self, text: &str) -> Result<Option<u64>, LlamaCppError> {
        let url = format!("{}/tokenize", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "content": text }))
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let body = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        Ok(counted_tokens_from_response(&body, text))
    }

    /// Query token-generation timing and usage for a model from the native
    /// `/props` endpoint. Returns `None` when the server does not expose it.
    pub async fn timings(&self) -> Result<Option<LlamaCppTimings>, LlamaCppError> {
        let url = format!("{}/props", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let props = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        let dgs = props
            .get("default_generation_settings")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let n_predict = dgs.get("n_predict").and_then(|v| v.as_u64()).unwrap_or(0);
        Ok(Some(LlamaCppTimings {
            tokens_generated: n_predict,
            ..LlamaCppTimings::default()
        }))
    }

    /// Check health of llama.cpp server and verify model availability if a model name is provided.
    pub async fn check_health(&mut self, model: &str) -> Result<ModelHealth, LlamaCppError> {
        let start = Instant::now();
        if model.is_empty() || model == "llamacpp" {
            return match self.ping().await {
                Ok(response_time) => {
                    let health = ModelHealth {
                        status: HealthStatus::Healthy,
                        response_time,
                        last_check: crate::health::current_timestamp(),
                        error_message: None,
                    };
                    self.health_tracker.update("llamacpp", health.clone());
                    Ok(health)
                }
                Err(e) => {
                    let health = ModelHealth {
                        status: HealthStatus::Unavailable,
                        response_time: start.elapsed().as_secs_f64(),
                        last_check: crate::health::current_timestamp(),
                        error_message: Some(e.to_string()),
                    };
                    self.health_tracker.update("llamacpp", health.clone());
                    Ok(health)
                }
            };
        }

        match self.list_models().await {
            Ok(models) => {
                let response_time = start.elapsed().as_secs_f64();
                let matches_model = models.iter().any(|m| {
                    m.id == model
                        || m.id.ends_with(&format!("/{model}"))
                        || m.id.split('/').next_back() == Some(model)
                });
                if matches_model {
                    let health = ModelHealth {
                        status: HealthStatus::Healthy,
                        response_time,
                        last_check: crate::health::current_timestamp(),
                        error_message: None,
                    };
                    self.health_tracker.update(model, health.clone());
                    Ok(health)
                } else {
                    let health = ModelHealth {
                        status: HealthStatus::Unavailable,
                        response_time,
                        last_check: crate::health::current_timestamp(),
                        error_message: Some(format!(
                            "model '{model}' is not loaded on llama.cpp server"
                        )),
                    };
                    self.health_tracker.update(model, health.clone());
                    Ok(health)
                }
            }
            Err(e) => {
                let health = ModelHealth {
                    status: HealthStatus::Unavailable,
                    response_time: start.elapsed().as_secs_f64(),
                    last_check: crate::health::current_timestamp(),
                    error_message: Some(e.to_string()),
                };
                self.health_tracker.update(model, health.clone());
                Ok(health)
            }
        }
    }

    /// Get configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Start a `llama-server` process hosting the given GGUF model.
///
/// Returns a handle to the spawned process. The process keeps running until
/// [`LlamaServerProcess::stop`] is called (or the child exits on its own).
pub fn start_llama_server(
    executable: &str,
    model_path: &str,
    port: u16,
    extra_args: &[&str],
) -> Result<LlamaServerProcess, LlamaCppError> {
    let mut cmd = std::process::Command::new(executable);
    cmd.arg("--host").arg("127.0.0.1");
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_args);
    cmd.stdout(Stdio::null());
    // What a server says while failing is the whole difference between
    // "it did not become ready" and "the device ran out of memory". Discarding
    // it — which is what sending this to the null device did — left every
    // launch in this program able to report only that something did not happen.
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| LlamaCppError::Api(format!("failed to start llama-server: {e}")))?;
    let log = ServerLog::default();
    let mut reader = None;
    if let Some(stderr) = child.stderr.take() {
        let filled = log.clone();
        match std::thread::Builder::new()
            .name("llama-server-log".to_string())
            .spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    filled.push(line);
                }
            }) {
            Ok(handle) => reader = Some(handle),
            Err(e) => {
                // A server whose failures cannot be read is a server that fails
                // quietly, which is the thing this whole path exists to prevent,
                // so stop the one just started instead of handing it back.
                let _ = child.kill();
                let _ = child.wait();
                return Err(LlamaCppError::Api(format!(
                    "failed to watch llama-server output: {e}"
                )));
            }
        }
    }

    Ok(LlamaServerProcess {
        child,
        base_url: format!("http://127.0.0.1:{}", port),
        log,
        reader,
    })
}

/// Assemble the command line for a server xencode starts itself.
///
/// Order is the whole point: the profile's preset first, then the model alias,
/// then whatever the user put in `llama_cpp_args` — last. `llama-server` runs
/// with the final value given for a flag, measured on b10809 by starting one
/// with `--ctx-size 8192 --ctx-size 2048`: `/props` reported 2048. Anything
/// earlier would let a preset quietly win an argument the user wrote down.
pub fn server_launch_args(
    profile_args: &[String],
    alias: Option<&str>,
    user_args: &[String],
) -> Vec<String> {
    let mut args = profile_args.to_vec();
    if let Some(alias) = alias {
        if !alias.trim().is_empty() && !args.iter().any(|a| a == "--alias") {
            args.push("--alias".to_string());
            args.push(alias.trim().to_string());
        }
    }
    args.extend(user_args.iter().cloned());
    args
}

/// The launch flags for the reasoning setting, or the one sentence explaining
/// why the setting says nothing usable.
///
/// `None`, a blank string and `auto` all say nothing: the model's own chat
/// template decides whether it thinks, which is what every build did before this
/// setting existed. `off` becomes `--reasoning off`; a whole number becomes
/// `--reasoning-budget <n>`.
///
/// **Why this is a launch flag and not a request field.** Measured on
/// `llama-server` b10809 with Qwen3-0.6B-Q4_K_M, asking one question at
/// temperature 0: the same request sent plain, sent with `reasoning_budget: 16`,
/// sent with `reasoning_effort: "minimal"`, and sent with
/// `chat_template_kwargs: {"thinking": false}`. The three that carried a field
/// came back identical to each other — 681 completion tokens, 1981 characters of
/// thinking — so a 16-token budget and an instruction not to think both did
/// nothing. Separately, a request carrying a key that exists nowhere in
/// llama.cpp was answered with HTTP 200 and no log line, which is the general
/// problem: a server accepting a field is not evidence that it read the field.
/// What does bite is the command line, measured the same way on the same model
/// and question:
///
/// | launched with | thinking characters | answer characters |
/// |---|---|---|
/// | (nothing) | 1352 | 354 |
/// | `--reasoning-budget 32` | 98 | 871 |
/// | `--reasoning-budget 0` | 0 | 2577 |
/// | `--reasoning off` | 0 | 358 |
///
/// Read the last two rows together, because they are the trap in this setting:
/// a budget of `0` is not the same as turning thinking off. It enters the
/// thinking block, ends it immediately, and the model goes on to write its
/// reasoning into the answer instead — 2577 characters of it for a one-number
/// question, against 358 from `--reasoning off`. A budget too small to finish a
/// chain does not fail and does not warn; the answer simply comes from a
/// half-finished plan. On this one question the unrestricted run and the
/// `--reasoning-budget 32` run both reached for the wrong number while `off` got
/// it right, which is a single sample on a 0.6B model and is recorded as that,
/// not as a rule about budgets.
///
/// The `/props` endpoint does not echo either flag, so unlike the context window
/// and the slot count there is no read-back: what was asked is visible in the
/// command line the program prints, and the effect is only visible in what comes
/// back.
pub fn reasoning_launch_args(setting: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = setting.map(str::trim) else {
        return Ok(Vec::new());
    };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Ok(Vec::new());
    }
    if raw.eq_ignore_ascii_case("off") {
        return Ok(vec!["--reasoning".to_string(), "off".to_string()]);
    }
    match raw.parse::<i64>() {
        Ok(budget) if budget >= 0 => Ok(vec![
            "--reasoning-budget".to_string(),
            budget.to_string(),
        ]),
        _ => Err(format!(
            "llama_cpp_reasoning must be \"auto\", \"off\" or a token budget like \"256\", not {raw:?}"
        )),
    }
}

/// Resolve the `llama-server` binary; tries the explicit path supplied by the
/// user, then common names on `PATH`.
pub fn find_llama_server(executable: Option<&str>) -> Option<String> {
    if let Some(exe) = executable {
        if !exe.trim().is_empty() {
            return Some(exe.trim().to_string());
        }
    }
    for candidate in [
        "llama-server",
        "llama-server.exe",
        "llama_server",
        "llama_cli",
    ] {
        if lookup_in_path(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn lookup_in_path(name: &str) -> bool {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let full = dir.join(name);
            if full.is_file() {
                return true;
            }
        }
    }
    false
}

/// Home-based directories that plausibly hold GGUF models, in preference order.
fn candidate_model_dirs(home: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = vec![
        home.join(".cache").join("llama.cpp"),
        home.join(".llama").join("models"),
        home.join(".local")
            .join("share")
            .join("llama.cpp")
            .join("models"),
    ];
    dirs.push(home.join("models"));
    dirs.push(home.join("models").join("llama.cpp"));
    dirs.push(std::path::PathBuf::from("models"));
    dirs
}

/// The user's home directory, tolerating missing env vars.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("USERPROFILE")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE")?;
            let path = std::env::var_os("HOMEPATH")?;
            Some(
                std::path::PathBuf::from(drive.to_string_lossy().into_owned())
                    .join(path.to_string_lossy().into_owned()),
            )
        })
        .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from))
}

/// Discover a GGUF model file to host when `llama_cpp_model_path` is empty.
///
/// Prefers an explicit path; otherwise scans the standard llama.cpp model
/// locations. `hint_name` (e.g. the `qwen3-4b` part of a `llamacpp:qwen3-4b`
/// model id) disambiguates when several GGUFs are present.
pub fn resolve_gguf_model(explicit: Option<&str>, hint_name: Option<&str>) -> Option<String> {
    let dirs = home_dir()
        .map(|h| candidate_model_dirs(&h))
        .unwrap_or_default();
    resolve_gguf_model_in(explicit, hint_name, &dirs)
}

/// Core of [`resolve_gguf_model`], parameterised over candidate directories so
/// it is testable without touching the real home directory.
fn resolve_gguf_model_in(
    explicit: Option<&str>,
    hint_name: Option<&str>,
    dirs: &[std::path::PathBuf],
) -> Option<String> {
    if let Some(explicit) = explicit {
        if !explicit.trim().is_empty() {
            return Some(explicit.trim().to_string());
        }
    }

    // Collect *.gguf files from each candidate dir, plus one level of subdirs
    // (common layout: `models/<model-name>/<model>.gguf`). The match key keeps
    // the containing folder name so a hint can match on either.
    let mut found: Vec<(std::path::PathBuf, String)> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_gguf = |p: &std::path::Path| {
                p.is_file()
                    && p.extension()
                        .map(|e| e.eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
            };
            if is_gguf(&path) {
                let key = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                found.push((path, key));
                continue;
            }
            if path.is_dir() {
                let folder_key = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                if let Ok(inner) = std::fs::read_dir(&path) {
                    for child in inner.flatten() {
                        let child_path = child.path();
                        if is_gguf(&child_path) {
                            let stem = child_path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_ascii_lowercase();
                            let key = if folder_key.contains(&stem) || stem.contains(&folder_key) {
                                folder_key.clone()
                            } else {
                                format!("{folder_key}/{stem}")
                            };
                            found.push((child_path, key));
                        }
                    }
                }
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    if let Some(hint) = hint_name {
        let hint_lower = hint.to_ascii_lowercase();
        if let Some(matched) = found.iter().find(|(_, key)| key.contains(&hint_lower)) {
            return Some(matched.0.to_string_lossy().into_owned());
        }
    }
    if found.len() == 1 {
        return Some(found[0].0.to_string_lossy().into_owned());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_client_has_correct_url() {
        let client = LlamaCppClient::default_client();
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn custom_client_trims_trailing_slash() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080/", 15);
        assert_eq!(client.base_url(), "http://127.0.0.1:8080");
    }

    #[test]
    fn options_default_is_empty() {
        let opts = LlamaCppOptions::default();
        assert!(opts.grammar.is_none());
        assert!(opts.json_schema.is_none());
        assert!(opts.min_p.is_none());
    }

    /// The record a completion response's `usage` produces: what the server
    /// counted, what it reused, and what is left over as the work it actually did.
    #[test]
    fn timings_keep_the_servers_counts_and_derive_the_work() {
        // The numbers from a real b10809 reply: 3,202 prompt tokens, of which
        // 3,201 were already in memory.
        let t = LlamaCppTimings::from_usage(100, 3202, 3201, 2.0);
        assert_eq!(t.tokens_generated, 100);
        assert_eq!(t.prompt_tokens, 3202);
        assert_eq!(t.cached_tokens, 3201);
        assert_eq!(
            t.tokens_evaluated, 1,
            "a reused prefix is not work the server repeated"
        );
        assert_eq!(t.predicted_per_second, 50.0);
        assert_eq!(t.total_seconds, 2.0);
        // Nothing is claimed about prompt speed that the response does not say.
        assert_eq!(t.prompt_per_second, 0.0);

        let cold = LlamaCppTimings::from_usage(8, 3202, 1, 4.0);
        assert_eq!(cold.tokens_evaluated, 3201, "a first prompt is all work");
    }

    /// A server that reports more reuse than prompt is nonsense rather than a
    /// panic or a wrap-around into a huge evaluation count.
    #[test]
    fn timings_survive_no_elapsed_time_and_too_much_reuse() {
        let t = LlamaCppTimings::from_usage(100, 0, 0, 0.0);
        assert_eq!(t.predicted_per_second, 0.0);
        assert_eq!(t.tokens_evaluated, 0);

        let impossible = LlamaCppTimings::from_usage(10, 5, 500, 1.0);
        assert_eq!(impossible.tokens_evaluated, 0, "{impossible:?}");
    }

    #[test]
    fn find_llama_server_prefers_explicit_path() {
        assert_eq!(
            find_llama_server(Some("C:\\tools\\llama-server.exe")),
            Some("C:\\tools\\llama-server.exe".to_string())
        );
        // Whitespace-only explicit path behaves exactly like no explicit path —
        // independent of whether llama-server happens to be present on PATH.
        assert_eq!(find_llama_server(Some("  ")), find_llama_server(None));
    }

    #[test]
    fn resolve_gguf_prefers_explicit_path() {
        assert_eq!(
            resolve_gguf_model_in(
                Some("D:\\models\\qwen3-4b.gguf"),
                None,
                &[std::path::PathBuf::from("C:\\nonesuch")]
            ),
            Some("D:\\models\\qwen3-4b.gguf".to_string())
        );
    }

    #[test]
    fn resolve_gguf_hint_disambiguates_multiple_files() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("llama3.2.gguf"), b"x").unwrap();
        std::fs::write(dir.join("qwen3-4b.gguf"), b"x").unwrap();

        let found = resolve_gguf_model_in(None, Some("qwen3-4b"), std::slice::from_ref(&dir));
        assert_eq!(
            found,
            Some(dir.join("qwen3-4b.gguf").to_string_lossy().into_owned())
        );

        // No hint + multiple candidates is ambiguous.
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_hint_matches_nested_model_folder() {
        // Mirrors the real layout: `~/models/<model>/<model>.gguf`.
        let dir = std::env::temp_dir().join(format!("xencode-gguf-nest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("llama3.2")).unwrap();
        std::fs::create_dir_all(dir.join("qwen3-4b")).unwrap();
        std::fs::write(dir.join("llama3.2").join("llama3.2-Q4_K_M.gguf"), b"x").unwrap();
        std::fs::write(dir.join("qwen3-4b").join("Qwen3-4B-Q4_K_M.gguf"), b"x").unwrap();

        assert_eq!(
            resolve_gguf_model_in(None, Some("qwen3-4b"), std::slice::from_ref(&dir)),
            Some(
                dir.join("qwen3-4b")
                    .join("Qwen3-4B-Q4_K_M.gguf")
                    .to_string_lossy()
                    .into_owned()
            )
        );
        // Ambiguous without a hint (one flat file + two nested families found).
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_single_file_without_hint() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-single-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("only.gguf"), b"x").unwrap();

        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            Some(dir.join("only.gguf").to_string_lossy().into_owned())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_no_files_returns_none() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-empty-{}", std::process::id()));
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
    }

    #[test]
    fn timings_defaults_zero() {
        let t = LlamaCppTimings::default();
        assert_eq!(t.tokens_generated, 0);
        assert_eq!(t.predicted_per_second, 0.0);
        assert_eq!(t.total_seconds, 0.0);
    }

    /// `/props` as answered by `llama-server` b10809 started with `-c 8192`,
    /// captured from the running server on this machine and trimmed to the keys
    /// that matter here. Note what is absent: there is no top-level `n_ctx`.
    fn props_b10809() -> serde_json::Value {
        serde_json::json!({
            "bos_token": "<|endoftext|>",
            "build_info": "b10809-5266f24da7",
            "chat_template": "{%- if tools %}",
            "default_generation_settings": {
                "n_ctx": 8192,
                "params": { "n_predict": -1, "temperature": 0.8, "top_k": 40 }
            },
            "endpoint_props": false,
            "eos_token": "<|im_end|>",
            "model_alias": "dolphin",
            "model_ftype": "Q4_K - Medium",
            "model_path": "/models/Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf",
            "total_slots": 4
        })
    }

    #[test]
    fn context_window_is_read_from_the_generation_settings() {
        assert_eq!(context_window_from_props(&props_b10809()), Some(8192));
    }

    #[test]
    fn context_window_falls_back_to_a_top_level_report() {
        let props = serde_json::json!({ "n_ctx": 4096, "total_slots": 1 });
        assert_eq!(context_window_from_props(&props), Some(4096));
    }

    #[test]
    fn context_window_is_absent_when_the_server_does_not_report_one() {
        let empty = serde_json::json!({ "build_info": "b10809", "total_slots": 4 });
        assert_eq!(context_window_from_props(&empty), None);
        let zero = serde_json::json!({ "default_generation_settings": { "n_ctx": 0 } });
        assert_eq!(context_window_from_props(&zero), None);
        let text = serde_json::json!({ "default_generation_settings": { "n_ctx": "8192" } });
        assert_eq!(context_window_from_props(&text), None);
        let huge = serde_json::json!({ "n_ctx": 5_000_000_000_u64 });
        assert_eq!(context_window_from_props(&huge), None);
    }

    #[test]
    fn a_usable_nested_window_beats_an_unusable_top_level_one() {
        let props = serde_json::json!({
            "n_ctx": 0,
            "default_generation_settings": { "n_ctx": 2048 }
        });
        assert_eq!(context_window_from_props(&props), Some(2048));
    }

    /// `/tokenize` as answered by `llama-server` b10809 to the prose sentence
    /// `"The context budget now knows the window it is spending."`, captured
    /// from the running server on this machine.
    #[test]
    fn a_token_count_is_the_length_of_the_token_array() {
        let body = serde_json::json!({
            "tokens": [785, 2266, 8039, 1431, 8788, 279, 3241, 432, 374, 10164, 13]
        });
        assert_eq!(
            counted_tokens_from_response(&body, "The context budget now knows…"),
            Some(11)
        );
    }

    #[test]
    fn a_response_without_a_token_array_is_not_an_answer() {
        for body in [
            serde_json::json!({}),
            serde_json::json!({ "token_count": 11 }),
            serde_json::json!({ "tokens": "785 2266" }),
            serde_json::json!({ "error": "unsupported" }),
        ] {
            assert_eq!(counted_tokens_from_response(&body, "some text"), None);
        }
    }

    /// Zero tokens is a real answer about an empty string and a fake answer
    /// about anything else: b10809 replies `{"tokens":[]}` with HTTP 200 to a
    /// request whose field name it does not read.
    #[test]
    fn zero_tokens_is_only_believed_for_text_that_was_actually_empty() {
        let empty = serde_json::json!({ "tokens": [] });
        assert_eq!(counted_tokens_from_response(&empty, ""), Some(0));
        assert_eq!(counted_tokens_from_response(&empty, "hello"), None);
    }

    /// Live check against a running `llama-server` — needs a real server, so it
    /// is skipped by default. Point it at one with
    /// `XENCODE_TEST_LLAMA_URL=http://127.0.0.1:8099 cargo test -p
    /// xencode-models-rs -- --ignored`, and start that server with a window you
    /// know, since the assertion is only "it reported one".
    #[tokio::test]
    #[ignore]
    async fn a_running_server_reports_its_own_window() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let reported = LlamaCppClient::new(&url, 5)
            .context_window()
            .await
            .expect("server did not answer /props");
        let tokens = reported.expect("server reported no window");
        assert!(tokens > 0, "reported window {tokens} is not usable");
        println!("{url} is running a {tokens}-token window");
    }

    /// The same server counting a sentence it is given, with the arithmetic the
    /// budgeter would have done instead (`ceil(chars / 4)`) printed beside it —
    /// the comparison is the reason to ask. Needs a real server, so it is skipped
    /// by default; see [`a_running_server_reports_its_own_window`] for the URL
    /// variable.
    #[tokio::test]
    #[ignore]
    async fn a_running_server_counts_the_text_it_is_given() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let text = "The context budget now knows the window it is spending.";
        let counted = LlamaCppClient::new(&url, 5)
            .count_tokens(text)
            .await
            .expect("server did not answer /tokenize");
        let tokens = counted.expect("server returned no usable count for non-empty text");
        let chars = text.len();
        let estimated = chars.div_ceil(4) as u64;
        assert!(tokens > 0, "counted {tokens} tokens for {chars} characters");
        println!("{url}: {chars} characters = {tokens} tokens counted, {estimated} estimated");
    }

    /// Waiting costs the wait and nothing else: a server that never answers
    /// leaves the report empty rather than hanging, so the caller can still say
    /// "unverified" and move on. Nothing listens on port 9 here, so the requests
    /// are refused on the loopback interface.
    #[tokio::test]
    async fn waiting_for_a_server_that_never_answers_gives_up_empty() {
        let client = LlamaCppClient::new("http://127.0.0.1:9", 1);
        let report = client
            .report_when_ready(2, std::time::Duration::from_millis(10))
            .await;
        assert_eq!(report, ServerReport::default());
    }

    /// The slot count beside the window, from the same `/props` request. Needs a
    /// real server; see [`a_running_server_reports_its_own_window`] for the URL
    /// variable.
    #[tokio::test]
    #[ignore]
    async fn a_running_server_reports_its_slot_count() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let report = LlamaCppClient::new(&url, 5)
            .report()
            .await
            .expect("server did not answer /props");
        let slots = report.slots.expect("server reported no slot count");
        assert!(slots > 0, "reported {slots} slots");
        println!(
            "{url}: {slots} slot(s), {} tokens of context each",
            report.context_tokens.unwrap_or(0)
        );
    }

    /// A flag written twice is not a conflict — `llama-server` runs with the last
    /// value, measured on b10809 by starting one with `--ctx-size 8192
    /// --ctx-size 2048` and reading 2048 back from `/props`. That is what makes
    /// the ordering here load-bearing: the user's own arguments have to be the
    /// later ones or the preset would overrule the person who wrote them.
    #[test]
    fn what_the_user_configured_is_the_last_word() {
        let args = server_launch_args(
            &["--ctx-size".to_string(), "8192".to_string()],
            Some("tiny"),
            &["--ctx-size".to_string(), "32768".to_string()],
        );
        assert_eq!(
            args,
            [
                "--ctx-size",
                "8192",
                "--alias",
                "tiny",
                "--ctx-size",
                "32768",
            ]
        );
    }

    /// An alias already in the preset is not said a second time, and a blank one
    /// is not said at all — `llama-server` takes `--alias` to mean "the next
    /// argument is the name", so a bare one would swallow the flag after it.
    #[test]
    fn an_alias_is_said_once_and_never_blank() {
        let args = server_launch_args(
            &["--alias".to_string(), "preset".to_string()],
            Some("tiny"),
            &[],
        );
        assert_eq!(args, ["--alias", "preset"]);

        let added = server_launch_args(
            &["--ctx-size".to_string(), "4096".to_string()],
            Some("tiny"),
            &[],
        );
        assert_eq!(added, ["--ctx-size", "4096", "--alias", "tiny"]);

        let none = server_launch_args(&["--ctx-size".to_string(), "4096".to_string()], None, &[]);
        assert_eq!(none, ["--ctx-size", "4096"]);

        let blank = server_launch_args(&[], Some("   "), &[]);
        assert!(blank.is_empty(), "{blank:?} carries a blank alias");
    }

    /// The three things the reasoning setting can mean, said as flags. `off` and
    /// a budget are different sentences — `--reasoning off` tells the template
    /// not to think at all, `--reasoning-budget N` lets it think for N tokens —
    /// so they cannot be folded into one "thinking: yes/no" switch.
    #[test]
    fn the_reasoning_setting_becomes_the_flag_that_means_it() {
        let cases: &[(&str, &[&str])] = &[
            ("off", &["--reasoning", "off"]),
            ("OFF", &["--reasoning", "off"]),
            ("256", &["--reasoning-budget", "256"]),
            ("0", &["--reasoning-budget", "0"]),
            ("  1024  ", &["--reasoning-budget", "1024"]),
            ("auto", &[]),
            ("", &[]),
        ];
        for (setting, want) in cases {
            let got = reasoning_launch_args(Some(setting)).expect("{setting} is a reasoning mode");
            assert_eq!(
                got.iter().map(String::as_str).collect::<Vec<_>>(),
                want.to_vec(),
                "for {setting:?}"
            );
        }
        assert!(reasoning_launch_args(None).unwrap().is_empty());
    }

    /// A value that names no mode is refused in the sentence a user can act on,
    /// rather than passed to `llama-server` — which accepts a flag it cannot
    /// parse by refusing to start at all, or (for a budget) by never hearing
    /// about it.
    #[test]
    fn a_reasoning_setting_that_names_nothing_is_refused_with_the_words() {
        for bad in ["-1", "lots", "256.5", "on", "1e3"] {
            let err = reasoning_launch_args(Some(bad))
                .err()
                .unwrap_or_else(|| panic!("{bad:?} was accepted as a reasoning setting"));
            assert!(
                err.contains("\"auto\", \"off\" or a token budget") && err.contains(bad),
                "refusal for {bad:?} does not say what is allowed: {err}"
            );
        }
    }

    /// The three answers a start-up check can give, in the words the user sees.
    #[test]
    fn a_server_either_agrees_disagrees_or_says_nothing() {
        let asked = ServerReport {
            context_tokens: Some(8192),
            slots: Some(1),
        };
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(8192),
                    slots: Some(1)
                }
            ),
            "BALANCED preset: 8192 tokens of context in 1 slot(s), as asked"
        );
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(2048),
                    slots: Some(1)
                }
            ),
            "BALANCED preset: 2048 tokens of context in 1 slot(s), not the 8192 tokens of context in 1 slot(s) asked for — later flags win, so check llama_cpp_args"
        );
        assert_eq!(
            settings_check_line("BALANCED preset", asked, ServerReport::default()),
            "BALANCED preset: the server reported no settings, so nothing is verified"
        );
        // A server that answers half of it has not contradicted the other half.
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(8192),
                    slots: None
                }
            ),
            "BALANCED preset: 8192 tokens of context, as asked"
        );
    }

    /// Shape read straight off a b10809 `/props` response for a server started
    /// with `--ctx-size 4096 --parallel 1`.
    #[test]
    fn a_server_says_its_window_and_its_slots() {
        let props = serde_json::json!({
            "default_generation_settings": { "n_ctx": 4096, "params": { "seed": 4294967295u64 } },
            "total_slots": 1,
            "build_info": "b10809-5266f24da7",
        });
        assert_eq!(
            report_from_props(&props),
            ServerReport {
                context_tokens: Some(4096),
                slots: Some(1),
            }
        );
    }

    /// A slot count that is absent, zero, or not a number says nothing. `0` is
    /// not "no parallelism" — it is a server that has not told us, and the caller
    /// has to be able to tell that apart from a real answer.
    #[test]
    fn a_server_that_does_not_report_its_slots_says_so_by_not_reporting_them() {
        for props in [
            serde_json::json!({ "default_generation_settings": { "n_ctx": 8192 } }),
            serde_json::json!({ "total_slots": 0 }),
            serde_json::json!({ "total_slots": "one" }),
            serde_json::json!({}),
        ] {
            assert_eq!(report_from_props(&props).slots, None, "{props}");
        }
    }

    /// A directory of its own for a test that writes an executable: the suite
    /// runs in parallel, and two tests pointed at one directory overwrite each
    /// other's script.
    fn scratch_dir(name: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(format!("xencode-llamacpp-{name}-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Write a program that behaves like a server with a bad launch: it says
    /// something on its error output and then ends. The process, the pipe, the
    /// exit code and the signal are the real ones — only the program is small.
    #[cfg(unix)]
    fn dying_server(script_body: &str) -> (std::path::PathBuf, String) {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch_dir("server");
        let path = dir.join("server");
        std::fs::write(&path, format!("#!/bin/sh\n{script_body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let executable = path.to_string_lossy().to_string();
        (dir, executable)
    }

    #[cfg(unix)]
    fn wait_for_exit(server: &mut LlamaServerProcess) -> ServerExit {
        for _ in 0..400 {
            if let Some(exit) = server.exit_status() {
                return exit;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("the process did not end within ten seconds");
    }

    #[test]
    fn a_death_by_signal_9_says_which_number_the_kernel_used() {
        // The number a shell reports for the same death is 137, which is 128 + 9;
        // this is the pair the plan's exit-137 signature refers to.
        let exit = ServerExit {
            code: None,
            signal: Some(9),
        };
        assert_eq!(
            exit.describe(),
            "killed by signal 9 — the kernel's out-of-memory killer sends this one"
        );
        assert!(exit.looks_like_out_of_memory(&[]));
        assert_eq!(
            ServerExit {
                code: Some(1),
                signal: None
            }
            .describe(),
            "exited with code 1"
        );
    }

    #[test]
    fn a_server_that_exits_cleanly_is_not_reported_as_a_failure() {
        let exit = ServerExit {
            code: Some(0),
            signal: None,
        };
        assert!(!exit.looks_like_out_of_memory(&[]), "{}", exit.describe());
    }

    #[test]
    fn the_ways_a_server_says_it_ran_out_are_all_recognised() {
        let exit = ServerExit {
            code: Some(1),
            signal: None,
        };
        for line in [
            "CUDA Error: out of memory at ggml-cuda.cu:1234",
            "terminate called after throwing an instance of 'std::bad_alloc'",
            "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
            "ggml_backend_cpu_buffer_from_buffer: failed to allocate buffer",
        ] {
            assert!(
                exit.looks_like_out_of_memory(&[line.to_string()]),
                "not recognised as a memory failure: {line}"
            );
        }
        // The two things a launch fails with that must not be retried as though
        // memory were the reason. The first is the interesting one: that line
        // appears on servers that are running fine, any time `--n-gpu-layers` is
        // pinned by the caller, so treating it as a failure would explain a
        // healthy server's unrelated death by the wrong thing.
        assert!(!exit.looks_like_out_of_memory(&[
            "error: invalid argument [o]: unknown option".to_string()
        ]));
        assert!(!exit.looks_like_out_of_memory(&[
            "W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99/-2, abort".to_string()
        ]));
    }

    #[cfg(unix)]
    #[test]
    fn what_the_server_printed_before_it_died_is_still_there_to_quote() {
        let (dir, executable) = dying_server(
            "echo 'ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory' >&2\nexit 1",
        );
        let mut server = start_llama_server(&executable, "unused.gguf", 0, &[]).unwrap();
        let exit = wait_for_exit(&mut server);
        assert_eq!(exit.code, Some(1));
        assert_eq!(exit.signal, None);
        let tail = server.log();
        assert!(
            tail.iter()
                .any(|line| line.contains("ErrorOutOfDeviceMemory")),
            "nothing was captured: {tail:?}"
        );
        assert!(exit.looks_like_out_of_memory(&tail));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_server_killed_by_a_signal_reports_the_signal_rather_than_a_code() {
        // Killing its own process group by name is how a shell reaches itself
        // with SIGKILL, which is the death the out-of-memory killer gives.
        let (dir, executable) = dying_server("kill -9 $$");
        let mut server = start_llama_server(&executable, "unused.gguf", 0, &[]).unwrap();
        let exit = wait_for_exit(&mut server);
        assert_eq!(exit.signal, Some(9), "{exit:?}");
        assert_eq!(exit.code, None);
        assert!(exit.looks_like_out_of_memory(&server.log()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn only_the_last_forty_lines_of_a_server_are_kept() {
        let (dir, executable) =
            dying_server("for i in $(seq 1 200); do echo \"line $i\" >&2; done");
        let mut server = start_llama_server(&executable, "unused.gguf", 0, &[]).unwrap();
        wait_for_exit(&mut server);
        // Reaping the process closes the pipe, and the reader stops with it, so
        // by the time it is done the whole run has passed through.
        for _ in 0..400 {
            if server.log().len() == LOG_TAIL_LINES {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let tail = server.log();
        assert_eq!(tail.len(), LOG_TAIL_LINES, "{tail:?}");
        assert_eq!(tail.last().map(String::as_str), Some("line 200"));
        assert!(
            !tail.iter().any(|line| line == "line 1"),
            "the oldest lines should have made room: {tail:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_running_server_has_not_exited_and_stop_leaves_nothing_behind() {
        let (dir, executable) = dying_server("sleep 30");
        let mut server = start_llama_server(&executable, "unused.gguf", 0, &[]).unwrap();
        assert!(server.exit_status().is_none());
        assert!(server.is_running());
        server.stop().unwrap();
        assert!(!server.is_running());
        assert_eq!(server.exit_status().map(|exit| exit.signal), Some(Some(9)));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A port nothing is listening on, chosen by the OS so two tests running at
    /// once do not answer each other.
    fn unused_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn waiting_for_a_dead_server_stops_when_it_dies_and_quotes_why() {
        let (dir, executable) =
            dying_server("echo 'ggml_backend: cannot allocate buffer' >&2; kill -9 $$");
        let mut server =
            start_llama_server(&executable, "unused.gguf", unused_port(), &[]).unwrap();
        let waited = std::time::Instant::now();
        // 400 tries at 50 ms is twenty seconds of patience. The point of the
        // test is that this never comes close to using it.
        let outcome = server
            .wait_until_ready(400, std::time::Duration::from_millis(50), &|| false)
            .await;
        assert!(
            waited.elapsed() < std::time::Duration::from_secs(5),
            "waited {:?} for a process that had already gone",
            waited.elapsed()
        );
        match outcome {
            ServerStart::Died { exit, tail } => {
                assert_eq!(exit.signal, Some(9), "{}", exit.describe());
                assert!(exit.looks_like_out_of_memory(&tail), "{tail:?}");
                assert!(
                    tail.iter()
                        .any(|line| line.contains("cannot allocate buffer")),
                    "{tail:?}"
                );
            }
            other => panic!("expected the death of the process, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_that_answers_is_ready_rather_than_slow() {
        // The answering socket is a real one on a real port; only the program
        // behind it is small, because what is under test here is what the wait
        // loop does with an answer, not what produces one.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let answer = std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut stream, _)) = listener.accept() {
                let mut bytes = [0u8; 512];
                let _ = stream.read(&mut bytes);
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                );
                let _ = stream.flush();
            }
        });
        let (dir, executable) = dying_server("sleep 30");
        let mut server = start_llama_server(&executable, "unused.gguf", port, &[]).unwrap();
        let outcome = server
            .wait_until_ready(200, std::time::Duration::from_millis(50), &|| false)
            .await;
        assert!(matches!(outcome, ServerStart::Ready), "{outcome:?}");
        server.stop().ok();
        let _ = answer.join();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_still_loading_its_model_is_not_ready_and_its_death_is_found() {
        // The reply this test's socket gives is the one measured from
        // `llama-server` b10809 two seconds into a real launch, and the line the
        // program prints is the one it wrote three seconds in when the key-value
        // cache did not fit. Reading the first as readiness is what let a launch
        // report success for a server that was about to die, so the wait has to
        // keep going past it and find the death.
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let answering = stop.clone();
        let answer = std::thread::spawn(move || {
            use std::io::{Read, Write};
            while !answering.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let mut bytes = [0u8; 512];
                        let _ = stream.read(&mut bytes);
                        let _ = stream.write_all(
                            b"HTTP/1.1 503 Service Unavailable\r\n\
                              connection: close\r\ncontent-length: 0\r\n\r\n",
                        );
                        let _ = stream.flush();
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        });
        let (dir, executable) = dying_server(
            "sleep 3
             echo 'ggml_vulkan: Device memory allocation of size 1052835840 failed.' >&2
             echo 'ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory' >&2
             echo 'llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache' >&2
             exit 1",
        );
        let mut server = start_llama_server(&executable, "unused.gguf", port, &[]).unwrap();
        let started = std::time::Instant::now();
        let outcome = server
            .wait_until_ready(400, std::time::Duration::from_millis(50), &|| false)
            .await;
        stop.store(true, Ordering::Relaxed);
        let _ = answer.join();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "waited {:?} on a server that had already gone",
            started.elapsed()
        );
        match outcome {
            ServerStart::Died { exit, tail } => {
                assert_eq!(exit.code, Some(1), "{}", exit.describe());
                assert!(exit.looks_like_out_of_memory(&tail), "{tail:?}");
                assert!(
                    tail.iter()
                        .any(|line| line.contains("ErrorOutOfDeviceMemory")),
                    "{tail:?}"
                );
            }
            other => panic!("a server that dies during the load is not {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_live_server_that_has_not_answered_yet_is_the_only_thing_called_slow() {
        let (dir, executable) = dying_server("sleep 30");
        let mut server =
            start_llama_server(&executable, "unused.gguf", unused_port(), &[]).unwrap();
        let outcome = server
            .wait_until_ready(3, std::time::Duration::from_millis(20), &|| false)
            .await;
        assert!(matches!(outcome, ServerStart::NotReady), "{outcome:?}");
        server.stop().ok();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn giving_up_on_a_server_leaves_no_process_behind() {
        let (dir, executable) = dying_server("sleep 30");
        let mut server =
            start_llama_server(&executable, "unused.gguf", unused_port(), &[]).unwrap();
        let pid = server.pid();
        let outcome = server
            .wait_until_ready(200, std::time::Duration::from_millis(50), &|| true)
            .await;
        assert!(matches!(outcome, ServerStart::Cancelled), "{outcome:?}");
        assert!(!server.is_running(), "PID {pid} outlived the wait");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A program that keeps count of how many times it has been started, in a
    /// file next to itself, so a test can tell "launched once" from "launched
    /// twice" instead of trusting the report.
    #[cfg(unix)]
    const COUNT_AND_OUT_OF_MEMORY: &str = "D=$(dirname \"$0\")\necho x >> \"$D/attempts\"\n\
        echo 'ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory' >&2\nexit 1";

    #[cfg(unix)]
    fn attempts(dir: &std::path::Path) -> usize {
        std::fs::read_to_string(dir.join("attempts"))
            .map(|text| text.lines().count())
            .unwrap_or(0)
    }

    #[test]
    fn the_context_size_in_force_is_the_last_one_written() {
        // `llama-server` runs with the final value given for a flag, measured on
        // b10809, so "what was asked" is the last of them — including one this
        // function appends itself when it retries.
        let args: Vec<String> = ["--ctx-size", "8192", "--flash-attn", "on"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(ctx_size_in(&args), Some(8_192));
        let overridden: Vec<String> = [
            "--ctx-size",
            "8192",
            "--ctx-size",
            "4096",
            "--n-gpu-layers",
            "all",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(ctx_size_in(&overridden), Some(4_096));
        let short: Vec<String> = ["-c", "2048"].iter().map(|s| s.to_string()).collect();
        assert_eq!(ctx_size_in(&short), Some(2_048));
        let equals: Vec<String> = ["--ctx-size=1024"].iter().map(|s| s.to_string()).collect();
        assert_eq!(ctx_size_in(&equals), Some(1_024));
        assert_eq!(ctx_size_in(&[]), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_model_that_only_fits_at_a_shorter_window_is_started_at_that_window() {
        // The whole point of the retry: the first launch dies the way an
        // out-of-memory launch dies, the second one is smaller, and the server
        // that comes up is the one handed back.
        let (dir, executable) = dying_server(
            "D=$(dirname \"$0\")\necho x >> \"$D/attempts\"\nn=$(wc -l < \"$D/attempts\")\n\
             if [ \"$n\" -lt 2 ]; then\n\
             \x20 echo 'ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory' >&2\n\
             \x20 exit 1\n\
             fi\nexec sleep 30",
        );
        // Something has to answer on the port for the second attempt to be
        // considered ready, and it answers the first attempt "no" because that
        // is what a server that has not loaded its model does.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let served_dir = dir.clone();
        let served = std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let started = std::fs::read_to_string(served_dir.join("attempts"))
                    .map(|text| text.lines().count())
                    .unwrap_or(0);
                let mut bytes = [0u8; 512];
                let _ = stream.read(&mut bytes);
                if started >= 2 {
                    let _ = stream.write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    );
                    let _ = stream.flush();
                }
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        });

        let args: Vec<String> = ["--ctx-size", "8192"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let step_down = |tokens: u64| {
            if tokens / 2 >= 1024 {
                Some(tokens / 2)
            } else {
                None
            }
        };
        let outcome = launch_and_wait(
            &executable,
            "unused.gguf",
            port,
            &args,
            Patience {
                tries: 40,
                gap: std::time::Duration::from_millis(50),
            },
            &|| false,
            &step_down,
        )
        .await;
        let LaunchOutcome::Started { mut server, notes } = outcome else {
            match outcome {
                LaunchOutcome::Failed { lines } => panic!("the retry never came up: {lines:?}"),
                LaunchOutcome::Cancelled => panic!("cancelled with nothing to cancel"),
                LaunchOutcome::Started { .. } => unreachable!(),
            }
        };
        assert_eq!(attempts(&dir), 2, "one death, one retry");
        assert_eq!(
            notes,
            vec!["llama-server ran out of memory at 8192 tokens; starting again at 4096."]
        );
        assert!(server.is_running());
        server.stop().ok();
        // The answering thread is left where it is: it holds the only socket on
        // that port and the test is over.
        drop(served);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn running_out_twice_says_so_once_quotes_the_server_and_names_the_next_step() {
        let (dir, executable) = dying_server(COUNT_AND_OUT_OF_MEMORY);
        let args: Vec<String> = ["--ctx-size", "8192", "--n-gpu-layers", "all"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let step_down = |tokens: u64| {
            if tokens / 2 >= 1024 {
                Some(tokens / 2)
            } else {
                None
            }
        };
        let outcome = launch_and_wait(
            &executable,
            "/tmp/unused.gguf",
            unused_port(),
            &args,
            Patience {
                tries: 40,
                gap: std::time::Duration::from_millis(50),
            },
            &|| false,
            &step_down,
        )
        .await;
        let LaunchOutcome::Failed { lines } = outcome else {
            panic!("a server that never comes up cannot be a start: {outcome:?}");
        };
        // Exactly one retry — a machine too small is not out-guessed by asking
        // it four more times.
        assert_eq!(attempts(&dir), 2, "{lines:?}");
        let text = lines.join("\n");
        assert!(text.contains("ran out of memory at 8192 tokens"), "{text}");
        assert!(
            text.contains("at 4096 tokens as well"),
            "the second death should say the smaller window died too: {text}"
        );
        assert!(
            text.contains("ErrorOutOfDeviceMemory"),
            "the server's own words belong in the report: {text}"
        );
        assert!(text.contains("/tmp/unused.gguf"), "{text}");
        assert!(text.contains("xencode hw probe"), "{text}");
        assert!(text.contains("xencode colab up"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_memory_death_with_no_window_named_is_not_guessed_at() {
        // Halving a number that was never written would be inventing it, and the
        // report would then claim a retry that no one asked for.
        let (dir, executable) = dying_server(COUNT_AND_OUT_OF_MEMORY);
        let outcome = launch_and_wait(
            &executable,
            "unused.gguf",
            unused_port(),
            &[],
            Patience {
                tries: 20,
                gap: std::time::Duration::from_millis(50),
            },
            &|| false,
            &|_| Some(1024),
        )
        .await;
        let LaunchOutcome::Failed { lines } = outcome else {
            panic!("expected the failure to be reported: {outcome:?}");
        };
        assert_eq!(attempts(&dir), 1, "{lines:?}");
        let text = lines.join("\n");
        assert!(text.contains("no context size was asked for"), "{text}");
        assert!(text.contains("ErrorOutOfDeviceMemory"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_death_that_is_not_about_memory_is_quoted_and_not_retried() {
        let (dir, executable) = dying_server(
            "D=$(dirname \"$0\")\necho x >> \"$D/attempts\"\n\
             echo 'error: unable to load model: bad magic' >&2\nexit 1",
        );
        let args: Vec<String> = ["--ctx-size", "8192"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let outcome = launch_and_wait(
            &executable,
            "unused.gguf",
            unused_port(),
            &args,
            Patience {
                tries: 20,
                gap: std::time::Duration::from_millis(50),
            },
            &|| false,
            &|tokens| Some(tokens / 2),
        )
        .await;
        let LaunchOutcome::Failed { lines } = outcome else {
            panic!("expected the failure to be reported: {outcome:?}");
        };
        assert_eq!(attempts(&dir), 1, "a broken file is not a memory problem");
        let text = lines.join("\n");
        assert!(text.contains("stopped before it answered"), "{text}");
        assert!(text.contains("unable to load model"), "{text}");
        assert!(
            !text.contains("starting again"),
            "the retry belongs to memory failures only: {text}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_server_that_stays_silent_is_reported_as_that_with_what_it_printed() {
        let (dir, executable) = dying_server(
            "D=$(dirname \"$0\")\necho x >> \"$D/attempts\"\n\
             echo 'waiting for the model to load' >&2\nexec sleep 30",
        );
        let args: Vec<String> = ["--ctx-size", "8192"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let outcome = launch_and_wait(
            &executable,
            "unused.gguf",
            unused_port(),
            &args,
            Patience {
                tries: 2,
                gap: std::time::Duration::from_millis(20),
            },
            &|| false,
            &|tokens| Some(tokens / 2),
        )
        .await;
        let LaunchOutcome::Failed { lines } = outcome else {
            panic!("a server that never answers is not a start: {outcome:?}");
        };
        assert_eq!(attempts(&dir), 1, "patience is not retried");
        let text = lines.join("\n");
        assert!(text.contains("has not answered"), "{text}");
        assert!(text.contains("waiting for the model to load"), "{text}");
        // The process is not left running behind the report.
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn check_health_distinguishes_loaded_model_from_unloaded() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for _ in 0..3 {
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf);
                    let body = r#"{"object":"list","data":[{"id":"/path/to/my-loaded-model.gguf","object":"model"}]}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });

        let mut client = LlamaCppClient::new(&format!("http://127.0.0.1:{port}"), 5);
        let h1 = client.check_health("my-loaded-model.gguf").await.unwrap();
        assert_eq!(h1.status, HealthStatus::Healthy);

        let h2 = client.check_health("TOTALLY-FAKE-MODEL").await.unwrap();
        assert_eq!(h2.status, HealthStatus::Unavailable);
        assert!(h2
            .error_message
            .as_ref()
            .unwrap()
            .contains("TOTALLY-FAKE-MODEL"));

        let h3 = client.check_health("").await.unwrap();
        assert_eq!(h3.status, HealthStatus::Healthy);

        handle.join().unwrap();
    }
}
