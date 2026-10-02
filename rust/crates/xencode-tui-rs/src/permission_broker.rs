//! The permission broker (`OR-3`).
//!
//! When xencode hands a task to an external worker (Claude, Codex, Gemini …),
//! something has to decide how much that worker may do on its own. Left to
//! themselves, workers are eager: a launch line can carry `--yolo` or
//! `--dangerously-skip-permissions` and the worker then edits files and runs
//! shells with no one watching. The broker is the layer that refuses to let
//! that happen. It has two jobs, both of which stay entirely inside xencode so
//! they can be checked without spending a call on any vendor:
//!
//! 1. **The launch line.** `plan_launch` builds the argv a worker would start
//!    with. It writes the approval control *xencode* chose for the configured
//!    mode, and it strips any approval flag the worker (or a config it was
//!    handed) tried to set for itself. The rule it enforces is that an approval
//!    flag appears in the launch line only because xencode put it there — a
//!    worker never widens its own leash. Where the vendor can route approvals
//!    back to us (Claude's `--permission-prompt-tool`, wired over the headless
//!    MCP server from `M-5`), the broker asks instead of pre-granting; where it
//!    cannot, the broker pre-grants the *lowest* sufficient mode rather than the
//!    widest one the flag supports.
//!
//! 2. **An approval request.** While a worker runs, it may ask "may I write
//!    this?". `PermissionBroker::answer` decides by calling the *same*
//!    [`classify`] gate the interactive tool loop uses, so a worker is never
//!    judged by a softer rule than the user's own tools are. A refusal is
//!    recorded and surfaced — it does not vanish — so a denied write shows up at
//!    xencode's gate instead of being silently skipped.
//!
//! The flag tokens are not invented here. They are the spellings `AR-3`'s
//! contract probe verifies against each agent's live `--help`
//! (`xencode-agents-rs/src/contract.rs`, the `approval` claim), plus Claude's
//! `--permission-prompt-tool`, named in the plan. The `--permission-mode`
//! *values* (`default`, `acceptEdits`, `bypassPermissions`) are Claude Code's
//! documented settings for that flag.

use std::path::Path;

use serde_json::{Map, Value};

use crate::agent_tools::{classify, ApprovalMode, Permission, ToolClass};

/// One tool call a worker asked to run, as it arrives at the broker.
#[derive(Debug, Clone)]
pub struct WorkerRequest {
    pub tool: String,
    pub args: Map<String, Value>,
}

impl WorkerRequest {
    /// A file-changing request for one path — the shape the done-when cares
    /// about, since a *write* is what must be visibly denied.
    pub fn write(path: &str) -> Self {
        let mut args = Map::new();
        args.insert("path".into(), Value::String(path.to_string()));
        args.insert("content".into(), Value::String("x".into()));
        Self {
            tool: "write_file".into(),
            args,
        }
    }
}

/// A write or action the broker refused, kept so the refusal is *visible* at
/// xencode's gate rather than swallowed. `what` names the call and, for a path
/// argument, where it tried to land.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub tool: String,
    pub what: String,
    pub reason: String,
}

/// The broker for one worker session. Its `mode` is fixed when the worker is
/// launched; the only thing that ever grows is the set of classes the *user*
/// approved for the session, and only through `grant` — a worker has no way to
/// call it.
#[derive(Debug, Clone)]
pub struct PermissionBroker {
    mode: ApprovalMode,
    granted: Vec<ToolClass>,
    refusals: Vec<Refusal>,
}

impl PermissionBroker {
    /// Start a broker that will grant the worker no more than `mode` allows.
    pub fn new(mode: ApprovalMode) -> Self {
        Self {
            mode,
            granted: Vec::new(),
            refusals: Vec::new(),
        }
    }

    pub fn mode(&self) -> ApprovalMode {
        self.mode
    }

    /// The refusals so far, in order. This is the record the done-when calls
    /// "visible at xencode's gate."
    pub fn refusals(&self) -> &[Refusal] {
        &self.refusals
    }

    /// Answer one worker approval request against xencode's own gate. The
    /// decision is exactly what [`classify`] returns for the same call the user
    /// would see — no separate, softer worker policy. A `Deny` is recorded (and
    /// stays recorded) so it is surfaced rather than skipped.
    pub fn answer(&mut self, root: &Path, request: &WorkerRequest) -> Permission {
        let decision = classify(root, &request.tool, &request.args, self.mode, &self.granted);
        if decision == Permission::Deny {
            // Only a denial that we have not already logged gets appended, so a
            // worker hammering the same refused path cannot bury the record under
            // duplicates while still being refused every time.
            let what = describe(request);
            let already = self
                .refusals
                .iter()
                .any(|r| r.tool == request.tool && r.what == what);
            if !already {
                self.refusals.push(Refusal {
                    tool: request.tool.clone(),
                    what,
                    reason: "outside the workspace or in a protected zone".to_string(),
                });
            }
        }
        decision
    }

    /// Record that the *user* (not a worker) approved a class for the rest of
    /// the session. A worker is given no handle to this method; widening always
    /// originates with the operator, which is the point of `never widen a mode
    /// on a worker's own authority`.
    pub fn grant(&mut self, class: ToolClass) {
        if !self.granted.contains(&class) {
            self.granted.push(class);
        }
    }
}

/// A short, human-readable account of what a refused request targeted, so the
/// refusal can be shown at the gate. Prefers a `path`/`cwd` argument.
fn describe(request: &WorkerRequest) -> String {
    for key in ["path", "cwd"] {
        if let Some(Value::String(v)) = request.args.get(key) {
            return v.clone();
        }
    }
    request.tool.clone()
}

// ── The launch line ─────────────────────────────────────────────────────────

/// Whether xencode hands the worker a way to ask, a narrow pre-grant, or full
/// autonomy. Chosen only from the configured mode and what the vendor supports;
/// never from anything the worker supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grant {
    /// Add no autonomy flag at all. The worker runs on its own prompting default
    /// — the strictest, and what an `Ask` worker without a prompt route gets.
    Nothing,
    /// Point the worker's approvals back at xencode (Claude: `--permission-
    /// prompt-tool <tool>` plus `--permission-mode default`). Used when the mode
    /// asks and the vendor can prompt and the caller supplied a real tool name.
    Prompt,
    /// Full autonomy (Claude `--permission-mode bypassPermissions`; the
    /// vendor's own all-yes flag). Only ever chosen when `mode` is `AllAllow`.
    Bypass,
}

/// Every approval-control flag token a roster agent is known to accept, drawn
/// from the `approval` markers `AR-3` verifies against each `--help`. A worker
/// that puts any of these in its own launch args is overruled: only xencode's
/// chosen control survives. Values that follow a flag (Claude's `--permission-
/// mode` value, for instance) are consumed with the flag.
pub const APPROVAL_FLAGS: &[&str] = &[
    "--permission-mode",
    "--permission-prompt-tool",
    "--allowedTools",
    "--dangerously-skip-permissions",
    "--ask-for-approval",
    "--sandbox",
    "--approval-mode",
    "--auto",
    "--auto-approve",
    "--yolo",
    "--force",
    "--mode",
    "--trust-all-tools",
    "--trust-tools",
];

/// Whether a vendor can route a permission prompt back to a caller. Today that
/// is Claude, whose `--permission-prompt-tool` is the integration the plan names;
/// the others only offer all-or-nothing autonomy flags.
pub fn supports_prompt(agent: &str) -> bool {
    matches!(agent, "claude")
}

/// The all-yes flag each non-prompting vendor uses, for the record of what the
/// broker refuses to emit below `AllAllow`. Kept to flags `AR-3` verified.
fn bypass_flag(agent: &str) -> Option<&'static str> {
    match agent {
        "crush" => Some("--yolo"),
        "agy" => Some("--dangerously-skip-permissions"),
        _ => None,
    }
}

/// Choose the grant for a launch, purely from xencode's side of the table.
///
/// * `AllAllow` → `Bypass`. This is the only mode that grants full autonomy,
///   and it is the operator's choice, never the worker's.
/// * `Ask` on a prompting vendor with a real prompt tool supplied → `Prompt`.
///   xencode would rather be asked than pre-widen a worker it can watch.
/// * anything else → `Nothing`. When a vendor cannot prompt and the mode is not
///   `AllAllow`, the lowest sufficient grant is no extra flag at all — the
///   worker keeps its own prompting default. We do not hand it `--yolo` merely
///   because that is the only flag it happens to have.
pub fn choose_grant(agent: &str, mode: ApprovalMode, prompt_tool: Option<&str>) -> Grant {
    match mode {
        ApprovalMode::AllAllow => Grant::Bypass,
        ApprovalMode::Ask if supports_prompt(agent) && prompt_tool.is_some() => Grant::Prompt,
        _ => Grant::Nothing,
    }
}

/// Build the launch argv for a worker.
///
/// `base` is the roster's `one_shot` command (with the `{prompt}` slot dropped,
/// since the prompt is not part of an approval decision). `worker_supplied` are
/// extra args the worker or its config asked for. Every approval-control token
/// in `worker_supplied` — and the value riding on it — is removed, then the one
/// control xencode chose for `mode` is appended. The result's approval surface
/// is therefore exactly what xencode decided, which is the clause this item has
/// to prove. It is a pure string transformation: it makes no claim that a
/// vendor was actually launched.
pub fn plan_launch(
    agent: &str,
    mode: ApprovalMode,
    base: &[String],
    worker_supplied: &[String],
    prompt_tool: Option<&str>,
) -> (Vec<String>, Grant) {
    let mut argv: Vec<String> = base.to_vec();

    // Overrule anything the worker tried to set for itself.
    let mut filtered: Vec<String> = Vec::with_capacity(worker_supplied.len());
    let mut skip_value_for = false;
    for token in worker_supplied {
        if skip_value_for {
            // This token is the argument to the flag we just dropped; drop it too.
            skip_value_for = false;
            continue;
        }
        if APPROVAL_FLAGS.contains(&token.as_str()) {
            // `--permission-mode`/`--approval-mode`/`--mode` take a value that
            // must not leak; a bare `--yolo` does not.
            skip_value_for = matches!(
                token.as_str(),
                "--permission-mode"
                    | "--approval-mode"
                    | "--mode"
                    | "--ask-for-approval"
                    | "--sandbox"
                    | "--permission-prompt-tool"
                    | "--allowedTools"
                    | "--trust-tools"
            );
            continue;
        }
        filtered.push(token.clone());
    }
    argv.extend(filtered);

    let grant = choose_grant(agent, mode, prompt_tool);
    match grant {
        Grant::Nothing => {}
        Grant::Prompt => {
            let tool = prompt_tool.expect("Prompt grant only chosen with a prompt tool present");
            argv.push("--permission-prompt-tool".into());
            argv.push(tool.to_string());
            argv.push("--permission-mode".into());
            argv.push("default".into());
        }
        Grant::Bypass => {
            if supports_prompt(agent) {
                argv.push("--permission-mode".into());
                argv.push("bypassPermissions".into());
            } else if let Some(flag) = bypass_flag(agent) {
                argv.push(flag.into());
            }
            // A vendor with neither a prompt path nor a recorded all-yes flag
            // gets no bypass token: there is nothing safe to emit, so the
            // worker simply runs and its calls still hit the approval gate.
        }
    }
    (argv, grant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn args_map(pairs: &[(&str, &str)]) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, val) in pairs {
            m.insert((*k).to_string(), Value::String((*val).to_string()));
        }
        m
    }

    // ── Clause 1: a real denied write is visible at the gate ──────────────

    /// A write that lands outside the workspace is denied by xencode's own
    /// gate, the refusal is recorded and shown, and — proven against the real
    /// filesystem — the file is never written.
    #[test]
    fn a_denied_write_is_recorded_at_the_gate_and_never_touches_disk() {
        let root = std::env::temp_dir().join(format!("xencode-broker-deny-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let outside = root.join("..").join("definitely-outside-xencode.rs");

        let mut broker = PermissionBroker::new(ApprovalMode::Ask);
        let request = WorkerRequest {
            tool: "write_file".into(),
            args: args_map(&[("path", outside.to_str().unwrap())]),
        };

        let decision = broker.answer(&root, &request);
        assert_eq!(
            decision,
            Permission::Deny,
            "an outside write must be denied"
        );

        // Visible, not skipped: the refusal is on the record with its target.
        assert_eq!(broker.refusals().len(), 1);
        assert_eq!(broker.refusals()[0].tool, "write_file");
        assert!(broker.refusals()[0]
            .what
            .ends_with("definitely-outside-xencode.rs"));

        // And real: because it was denied, the file does not exist. A broker
        // that only *claimed* to deny would let a caller write anyway; here the
        // refusal is the last word, so we assert the absence on disk.
        let canonical = std::fs::canonicalize(&root)
            .unwrap()
            .parent()
            .unwrap()
            .join("definitely-outside-xencode.rs");
        assert!(
            !canonical.exists(),
            "a denied write must not have hit the disk"
        );

        // The denial is idempotent-visible: repeat requests keep being refused
        // but do not bury the record under duplicates.
        let again = broker.answer(&root, &request);
        assert_eq!(again, Permission::Deny);
        assert_eq!(
            broker.refusals().len(),
            1,
            "the same refusal is not logged twice"
        );
    }

    /// The same file INSIDE the workspace is not denied by the path rule — the
    /// broker's refusal is about the boundary, not about refusing all writes.
    #[test]
    fn a_write_inside_the_workspace_is_asked_not_refused() {
        let root = std::env::temp_dir().join(format!("xencode-broker-ask-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let inside: PathBuf = root.join("notes.md");

        let mut broker = PermissionBroker::new(ApprovalMode::Ask);
        let decision = broker.answer(
            &root,
            &WorkerRequest {
                tool: "write_file".into(),
                args: args_map(&[("path", inside.to_str().unwrap())]),
            },
        );
        assert_eq!(
            decision,
            Permission::Ask,
            "an inside write prompts the user"
        );
        assert!(broker.refusals().is_empty(), "an Ask is not a refusal");
    }

    // ── Clause 2: the launch line carries only what xencode chose ─────────

    /// A worker that tries to launch itself with full autonomy is overruled:
    /// the flag disappears and no approval control survives except xencode's.
    #[test]
    fn a_worker_cannot_widen_its_own_authority_in_the_launch_line() {
        let base = s(&["claude", "-p"]);
        let worker_tried = s(&[
            "--permission-mode",
            "bypassPermissions",
            "--allowedTools",
            "Bash",
        ]);

        // xencode's mode is Ask with a prompt route; it did NOT choose bypass.
        let (argv, grant) = plan_launch(
            "claude",
            ApprovalMode::Ask,
            &base,
            &worker_tried,
            Some("mcp__xencode__approve"),
        );
        assert_eq!(grant, Grant::Prompt);
        let joined = argv.join(" ");
        assert!(
            !joined.contains("bypassPermissions"),
            "worker's bypass was not overruled: {joined}"
        );
        assert!(
            !joined.contains("--allowedTools"),
            "worker's allowedTools were not overruled: {joined}"
        );
        // The ONLY approval flags present are the ones xencode chose.
        for token in &argv {
            if APPROVAL_FLAGS.contains(&token.as_str()) {
                assert!(
                    matches!(
                        token.as_str(),
                        "--permission-prompt-tool" | "--permission-mode"
                    ),
                    "unexpected approval flag {token} survived into the launch line"
                );
            }
        }
        assert!(joined.contains("--permission-prompt-tool mcp__xencode__approve"));
    }

    /// The bypass value following `--permission-mode` is consumed with the flag,
    /// so a stripped flag cannot leave its dangerous value orphaned in argv.
    #[test]
    fn stripping_an_approval_flag_takes_its_value_with_it() {
        let base = s(&["crush"]);
        let worker_tried = s(&["--permission-mode", "acceptEdits"]);
        let (argv, grant) = plan_launch("crush", ApprovalMode::Ask, &base, &worker_tried, None);
        assert_eq!(grant, Grant::Nothing);
        assert!(
            !argv.iter().any(|t| t.contains("acceptEdits")),
            "value orphaned: {argv:?}"
        );
        assert!(
            !argv.iter().any(|t| t == "--permission-mode"),
            "flag survived: {argv:?}"
        );
    }

    /// A vendor with only an all-yes flag does not get it handed over below
    /// `AllAllow`: the lowest sufficient grant is no flag, not `--yolo`.
    #[test]
    fn a_non_prompting_vendor_is_not_pre_granted_full_autonomy_by_default() {
        let base = s(&["crush"]);
        let (argv, grant) = plan_launch("crush", ApprovalMode::Ask, &base, &[], None);
        assert_eq!(grant, Grant::Nothing);
        assert!(
            !argv.iter().any(|t| t == "--yolo"),
            "crush must not be given --yolo under Ask: {argv:?}"
        );

        let (argv, grant) = plan_launch("crush", ApprovalMode::AllAllow, &base, &[], None);
        assert_eq!(grant, Grant::Bypass);
        assert!(
            argv.iter().any(|t| t == "--yolo"),
            "AllAllow is the one mode that grants --yolo"
        );
    }

    /// Only `AllAllow` ever produces a bypass control, and when it does, it is
    /// xencode's own doing — the grant is chosen from the mode, not the worker.
    #[test]
    fn only_all_allow_produces_a_bypass() {
        for mode in [ApprovalMode::Ask, ApprovalMode::EditAllow] {
            let (argv, grant) = plan_launch(
                "claude",
                mode,
                &s(&["claude", "-p"]),
                &[],
                Some("mcp__x__a"),
            );
            assert_ne!(grant, Grant::Bypass, "{mode:?} must never bypass");
            assert!(!argv.iter().any(|t| t == "bypassPermissions"));
        }
        let (_, grant) = plan_launch(
            "claude",
            ApprovalMode::AllAllow,
            &s(&["claude", "-p"]),
            &[],
            None,
        );
        assert_eq!(grant, Grant::Bypass);
    }

    /// When a prompting vendor's caller has no prompt tool wired yet, the broker
    /// refuses to invent one and falls back to the strictest grant — it does not
    /// silently promote the worker to bypass just because a flag was available.
    #[test]
    fn a_prompt_vendor_without_a_prompt_tool_stays_strict() {
        let (argv, grant) = plan_launch(
            "claude",
            ApprovalMode::Ask,
            &s(&["claude", "-p"]),
            &[],
            None,
        );
        assert_eq!(grant, Grant::Nothing);
        assert!(
            !argv.iter().any(|t| t.starts_with("--permission")),
            "no control invented: {argv:?}"
        );
    }
}
