//! Keybinding help data (E3-02) for the `?`/`F1` overlay.
//!
//! Every entry here mirrors a real arm in `keymap.rs` — when you change a
//! binding there, change it here. The unit test guards the format.

use crate::focus::FocusArea;
use crate::theme::ThemeColors;
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

type Binding = (&'static str, &'static str);

/// The conversation (TD-3): the input is always ready, so these are the
/// keys there.
const CONVERSATION: &[Binding] = &[
    (
        "Enter",
        "send · while a turn runs, the message waits and goes after it",
    ),
    ("Alt+Enter / Ctrl+J", "new line"),
    (
        "Esc",
        "stop the running turn · twice: your last message back to edit",
    ),
    ("Ctrl+C", "clear the input · on an empty input, twice quits"),
    ("Ctrl+P", "open a view, a panel, a command or a setting"),
    ("Ctrl+B", "show or hide the side panel"),
    ("Ctrl+O", "unfold the latest tool output · again folds it"),
    ("Ctrl+L", "the Models view"),
    (
        "Ctrl+E",
        "the Files view (on an empty input; else end of line)",
    ),
    ("Shift+Tab", "approval mode: ask → auto-edit → plan"),
    ("Ctrl+Space", "Coding ⇄ Orchestrator mode"),
    ("?", "this help (on an empty input)"),
    ("@path", "mention a file · Tab puts the best match in"),
    (
        "!command",
        "run a shell command, asking first like the agent does",
    ),
    ("/command", "a slash command · Tab completes it"),
    ("Alt+Up / Alt+Down", "previous / next message you sent"),
    (
        "Ctrl+A / K / U / W",
        "start of line · delete to end / to start / the word",
    ),
];

/// Orchestrator mode (OM-3): keys go to the focused tile; these follow Ctrl+A.
const ORCHESTRATOR: &[Binding] = &[
    (
        "Ctrl+Space",
        "enter or leave the mode (the agents keep running)",
    ),
    ("Ctrl+A 0-9", "focus that tile (0 is the conversation)"),
    ("Ctrl+A arrows / h j k l", "focus the previous or next tile"),
    ("Ctrl+A z", "zoom the focused tile, again to go back"),
    ("Ctrl+A t", "draw every agent as a tile"),
    ("Ctrl+A n", "the picker: open more agents"),
    ("Ctrl+A x x", "stop the focused agent (its worktree stays)"),
    ("Ctrl+A Ctrl+A", "send Ctrl+A to the agent"),
];

pub(crate) const GLOBAL: &[Binding] = &[
    (
        "Ctrl+C / q q",
        "quit outside the conversation — Ctrl+C stops a running turn first",
    ),
    ("Ctrl+G", "refresh git status"),
    ("Ctrl+S", "save editor / git commit panel"),
    ("Ctrl+R", "code review (AI, current file)"),
    ("Ctrl+Y", "PR review dashboard"),
    ("Ctrl+K", "background tasks panel"),
    ("Ctrl+X", "the palette, as Ctrl+P"),
    ("Ctrl+B", "ByteBot, from a panel"),
    ("Ctrl+,", "settings"),
    ("Ctrl+W", "close panel → chat"),
    ("Ctrl+N", "agent stack overlay (Esc closes)"),
    (
        "Mouse",
        "the wheel scrolls; a click picks a pane in Files (off with Settings → Mouse Capture)",
    ),
    (
        "Ctrl+T",
        "toggle the terminal strip under the editor in Files",
    ),
    ("Ctrl+H", "run provider health check"),
];

const UNIVERSAL: &[Binding] = &[
    ("Tab", "in Files, tree ↔ editor; in a view, the next tab"),
    ("i or /", "edit chat input"),
    ("e", "edit code editor"),
    ("m", "model selector (r refresh, l load, u unload)"),
    ("s", "settings (panels that bind s keep it)"),
    ("Esc", "close the panel · back to the conversation"),
    ("? / F1", "this help"),
];

/// The agent approval prompt (I1-03) is modal: these are the only keys it
/// answers while it is on screen.
const APPROVAL: &[Binding] = &[
    ("y", "allow this call"),
    ("a", "allow · and everything like it this session"),
    ("n / Esc", "deny (the model is told it was denied)"),
    ("k / j", "scroll the diff or command preview"),
];

/// Commands intercepted by `submit_message` — keep in sync with SLASH_COMMANDS.
pub(crate) const COMMANDS: &[Binding] = &[
    ("/help", "this overlay — every key and command"),
    ("/init [abort|status]", "generate & control project docs"),
    (
        "/ctx …",
        "context engine: track/eval/kv/archive · fold→promote",
    ),
    ("/advise [filter]", "repository insights"),
    (
        "/impact <file>",
        "blast radius of one file (crates · files · churn)",
    ),
    (
        "/workers",
        "workers, tasks, graph, costs, logs and approvals — each figure names its source",
    ),
    (
        "/orchestrator on|off|status|agents|tasks|graph|logs|costs|permissions|inspect|retry|stop|attach",
        "the orchestrator's command surface: the fleet panel by section, the launch grants, a named \
         row's sources, and the terminal handover that only runs when there is a terminal here",
    ),
    ("/bytebot <task>", "autonomous task execution"),
    ("/plan [clear|accept|decline]", "expand, clear, or respond to proposed plan tasks"),
    (
        "/rewind [turns] [--force]",
        "undo the agent's file changes (session-only), refusing files edited by hand",
    ),
    (
        "/lesson [status|set <words>|approve|drop]",
        "the lesson a rewind or a run of failing checks drafted — blank until you write it, \
         and only your approval puts it in AGENTS.md",
    ),
    (
        "/gate [bugfix [paths] | off]",
        "refuse production edits until a reproduction test has been seen failing",
    ),
    (
        "/mcp [status|stop|read <server> <uri>|prompt <server> <name>]",
        "connect MCP servers, read their tools, documents and prompts",
    ),
    (
        "/spawn <task> [#branch]",
        "run a subagent in a fresh git worktree",
    ),
    (
        "/plugin [reload]",
        "show which plugins took effect; reload re-scans the plugin dir",
    ),
    (
        "/skills [reload]",
        "show which SKILL.md skills loaded from the user directory and the project; reload re-scans",
    ),
    (
        "/trace [turns]",
        "what the recent agent turns did: rounds, tools, tokens (local, asks no model)",
    ),
    (
        "/cost",
        "spend, tokens and speed from the records on disk (local, asks no model)",
    ),
    (
        "/doctor [env|deps]",
        "probe machine resources, GPUs, memory, and environment facts",
    ),
    (
        "/verify [skip...]",
        "run machine-checkable verification checklist (fmt, lint, test)",
    ),
    (
        "/hotspots [limit]",
        "rank files by churn, size, and bus factor",
    ),
    ("/agents", "inventory installed coding-agent CLIs on PATH"),
    ("/files", "the Files view: the tree and the editor"),
    ("/changes", "the Changes view: the diff, AI review, commit"),
    ("/models", "the Models view: pick a model, your models, health"),
    ("/project", "the Project view: insights, impact, analysis, security"),
    ("/settings", "the Settings view"),
    (
        "/trust [status|forget] [path]",
        "trust an AGENTS.md's exact bytes as instructions — this workspace's, or a \
         directory's such as src/auth/AGENTS.md — or report/withdraw",
    ),
    (
        "/egress [text]",
        "show where the next turn would send your prompt, and what redaction holds back, without sending it",
    ),
    ("/goto <destination>", "switch focus to destination by name"),
    ("/model [name]", "switch model, or list the ones found"),
    ("/level [1-4]", "progressive disclosure tier: 1 Core, 2 Workflow, 3 Advanced, 4 All"),
];

pub(crate) fn panel_bindings(focus: FocusArea) -> &'static [Binding] {
    use FocusArea::*;
    match focus {
        ChatInput => &[("PageUp / PageDown", "scroll the conversation")],
        FileExplorer => &[
            ("↑ ↓ / j k", "move selection"),
            ("Enter", "open file in editor"),
            ("Space", "attach/detach file"),
        ],
        CodeEditor => &[
            ("↑ ↓ / j k", "scroll"),
            ("e", "type edits"),
            ("Ctrl+S", "save file"),
        ],
        ModelSelector => &[
            ("↑ ↓ / j k", "select model"),
            ("Enter", "set default model"),
            ("r", "refresh list"),
            ("l / u", "llama.cpp load / unload"),
        ],
        Settings => &[
            ("↑ ↓ / j k", "move cursor"),
            ("← →", "change value / number"),
            ("Enter", "toggle, edit URL/number, or save"),
            ("Factory Reset + Enter twice", "factory reset"),
        ],
        CodeReview => &[("↑ ↓ / j k", "scroll review"), ("Enter", "start review")],
        ProjectAnalyzer => &[("Enter", "start / continue")],
        MultiLanguage => &[
            ("Enter / d", "walk the workspace for language totals"),
            ("Tab", "pick the field letters edit (From, To, Text)"),
            ("type / Backspace", "edit that field"),
            ("Enter", "translate the text with one model call"),
            ("Esc", "leave the field"),
        ],
        GitCommit => &[
            ("type", "commit message"),
            ("Enter", "git commit -am (async)"),
            ("Backspace", "delete char"),
        ],
        ByteBotPanel => &[
            ("type", "command"),
            ("Enter", "run command"),
            ("↑", "recall last command"),
        ],
        ProviderHealth => &[("↑ ↓ / j k", "scroll"), ("Ctrl+H", "run check")],
        SecurityAuditor => &[
            ("↑ ↓ / j k", "scroll"),
            ("Enter", "start scan"),
            ("Space", "cycle severity filter"),
            ("s", "toggle sort severity ↔ category"),
        ],
        CustomModels => &[
            ("↑ ↓ / j k", "select profile"),
            ("n", "new profile from the current session"),
            ("- +", "temperature down / up"),
            ("← →", "max tokens down / up"),
            ("Enter", "apply to the next turn"),
            ("s", "write profiles to config.json"),
            ("t", "one test request to the provider"),
        ],
        ReviewDashboard => &[
            ("↑ ↓ / j k", "select file"),
            ("Enter", "reload diffs"),
            ("b", "toggle base HEAD ↔ main"),
            ("u / d", "scroll diff ±10"),
        ],
        TaskManager => &[
            ("↑ ↓ / j k", "select task · scroll in detail view"),
            ("Enter", "open/close task output"),
            ("x", "stop selected task"),
            ("d", "remove finished task"),
        ],
        WorktreePanel => &[
            ("↑ ↓ / j k", "select worktree"),
            ("a", "add worktree (path, then branch)"),
            ("d", "remove selected worktree (y to confirm)"),
            ("r", "refresh worktree list"),
            ("Esc", "cancel prompt · close panel"),
        ],
        AdvisePanel => &[
            ("↑ ↓ / j k", "select finding · scroll in detail view"),
            ("Enter", "open/close detail for the selected finding"),
            ("o", "open the advised file in the editor"),
            ("r", "recompute insights from the live snapshot"),
            ("Esc", "back to list · close panel"),
        ],
        ImpactPanel => &[
            ("↑ ↓ / j k", "select a row · scroll in detail view"),
            (
                "Enter",
                "open/close detail: hops, the via paths, co-change count",
            ),
            ("→", "descend: re-root the query at the selected file"),
            ("←", "back to the previous target on the descend stack"),
            ("o", "open the selected consumer file in the editor"),
            ("r", "recompute this file's blast radius in place"),
            ("Esc", "back to tree · pop descend stack · close panel"),
        ],
        WorkerPanel => &[
            ("↑ ↓ / j k", "select a row · scroll its detail"),
            (
                "Enter",
                "open/close the record, event or file each figure was read from",
            ),
            (
                "r",
                "re-read the streams, the registry and the records on disk",
            ),
            ("Esc", "back to list · close panel"),
        ],
    }
}

/// Help screen lines: current panel first, then universal/global/editing.
pub fn help_lines(focus: FocusArea, theme: &ThemeColors) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    section(&mut lines, "This panel", panel_bindings(focus), theme);
    section(&mut lines, "The conversation", CONVERSATION, theme);
    views(&mut lines, theme);
    section(&mut lines, "Orchestrator mode", ORCHESTRATOR, theme);
    section(&mut lines, "Agent approval prompt", APPROVAL, theme);
    section(&mut lines, "Slash commands", COMMANDS, theme);
    section(&mut lines, "In a panel", UNIVERSAL, theme);
    section(&mut lines, "More keys", GLOBAL, theme);
    lines
}

/// The seven views, read from the registry so the list cannot drift.
fn views(out: &mut Vec<Line<'static>>, theme: &ThemeColors) {
    let rows: Vec<(String, &'static str)> = crate::screens::Screen::ALL
        .iter()
        .map(|s| (format!("Ctrl+P {}", s.command()), s.description()))
        .collect();
    out.push(Line::from(Span::styled(
        " Views".to_string(),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    )));
    for (key, desc) in rows {
        out.push(Line::from(vec![
            Span::styled(
                format!("   {key:<22}"),
                Style::default().fg(theme.message_assistant),
            ),
            Span::styled(desc, Style::default().fg(theme.fg)),
        ]));
    }
    out.push(Line::from(""));
}

fn section(out: &mut Vec<Line<'static>>, title: &str, rows: &[Binding], theme: &ThemeColors) {
    out.push(Line::from(Span::styled(
        format!(" {title}"),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    )));
    for (key, desc) in rows {
        out.push(Line::from(vec![
            Span::styled(
                format!("   {key:<22}"),
                Style::default().fg(theme.message_assistant),
            ),
            Span::styled(*desc, Style::default().fg(theme.fg)),
        ]));
    }
    out.push(Line::from(""));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_panel_has_bindings() {
        for f in FocusArea::ALL {
            assert!(!panel_bindings(f).is_empty(), "{f:?} has no help rows");
        }
    }

    #[test]
    fn commands_section_lists_every_slash_command() {
        for cmd in crate::app::SLASH_COMMANDS {
            assert!(
                COMMANDS.iter().any(|(key, _)| key.starts_with(cmd)),
                "{cmd} missing from the help commands section"
            );
        }
    }

    #[test]
    fn help_screen_has_all_sections() {
        let lines = help_lines(FocusArea::ChatInput, &ThemeColors::get("ocean"));
        let text: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect::<Vec<_>>()
            .join("\n");
        for section in [
            "This panel",
            "The conversation",
            "Views",
            "Slash commands",
            "In a panel",
            "More keys",
        ] {
            assert!(text.contains(section), "missing {section}");
        }
        assert!(text.contains("Ctrl+Y"));
        assert!(text.contains("scroll the conversation"));
    }
}
