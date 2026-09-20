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

const GLOBAL: &[Binding] = &[
    ("Ctrl+C / q", "quit"),
    ("Ctrl+G", "refresh git status"),
    ("Ctrl+E", "file explorer"),
    ("Ctrl+S", "save editor / git commit panel"),
    ("Ctrl+R", "code review (AI, current file)"),
    ("Ctrl+Y", "PR review dashboard"),
    ("Ctrl+K", "background tasks panel"),
    ("Ctrl+O", "worktree panel"),
    ("Ctrl+L", "insights panel"),
    ("Ctrl+F", "feature navigator"),
    ("Ctrl+D", "performance dashboard"),
    ("Ctrl+P", "project analyzer"),
    ("Ctrl+B", "ByteBot agent panel"),
    ("Ctrl+,", "settings"),
    ("Ctrl+W", "close panel → chat"),
    ("Ctrl+T", "toggle terminal strip"),
    ("Ctrl+H", "run provider health check"),
];

const UNIVERSAL: &[Binding] = &[
    ("Tab", "cycle explorer → editor → chat"),
    ("i or /", "edit chat input"),
    ("e", "edit code editor"),
    ("m", "model selector (r refresh, l load, u unload)"),
    ("s", "settings (panels that bind s keep it)"),
    ("Esc", "close popup / leave edit mode"),
    ("? / F1", "this help"),
];

const EDITING: &[Binding] = &[
    ("Enter", "send message"),
    ("Alt+Enter / Ctrl+J", "insert newline"),
    ("Alt+Up / Alt+Down", "recall previous / next prompt"),
    ("Esc", "back to normal mode"),
    ("Tab", "complete /command · else 4 spaces"),
    ("← → ↑ ↓ Home End", "move cursor"),
    ("Backspace/Del", "delete"),
];

/// Commands intercepted by `submit_message` — keep in sync with SLASH_COMMANDS.
const COMMANDS: &[Binding] = &[
    ("/init [abort|status]", "generate & control project docs"),
    (
        "/ctx …",
        "context engine: status/track/compact/eval/kv/archive",
    ),
    ("/advise [filter]", "repository insights"),
    ("/bytebot <task>", "autonomous task execution"),
];

fn panel_bindings(focus: FocusArea) -> &'static [Binding] {
    use FocusArea::*;
    match focus {
        ChatInput => &[("↑ ↓ / j k", "scroll chat")],
        FileExplorer => &[
            ("↑ ↓ / j k", "move selection"),
            ("Enter", "open file in editor"),
            ("Space", "attach/detach file"),
        ],
        CodeEditor => &[("↑ ↓ / j k", "scroll"), ("e", "type edits"), ("Ctrl+S", "save file")],
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
            ("row 13 + Enter", "factory reset"),
        ],
        CodeReview => &[
            ("↑ ↓ / j k", "scroll review"),
            ("Enter", "start review"),
        ],
        PerformanceDashboard | ProjectAnalyzer | CollaborationHub | PerformanceProfiler
        | MultiLanguage => &[("Enter", "start / continue")],
        VoiceInterface => &[
            ("Enter", "start voice session"),
            ("Space / m", "toggle mute"),
        ],
        TerminalAssistant => &[
            ("Enter", "start assistant"),
            ("Space", "cycle risk filter"),
        ],
        GitCommit => &[
            ("type", "commit message"),
            ("Enter", "git commit -am (async)"),
            ("Backspace", "delete char"),
        ],
        FeatureNavigator => &[
            ("↑ ↓ / j k", "select feature"),
            ("Enter", "open feature"),
        ],
        ByteBotPanel => &[
            ("type", "command"),
            ("Enter", "run command"),
            ("↑", "recall last command"),
        ],
        ProviderHealth => &[
            ("↑ ↓ / j k", "scroll"),
            ("Ctrl+H", "run check"),
        ],
        SecurityAuditor => &[
            ("↑ ↓ / j k", "scroll"),
            ("Enter", "start scan"),
            ("Space", "cycle severity filter"),
            ("s", "toggle sort severity ↔ category"),
        ],
        CustomModels => &[
            ("↑ ↓ / ← →", "select profile"),
            ("Enter", "open profile editor"),
            ("s", "save profile (while editing)"),
        ],
        LearningMode => &[
            ("Enter", "start lesson / answer quiz"),
            ("← →", "pick quiz option"),
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
    }
}

/// Help screen lines: current panel first, then universal/global/editing.
pub fn help_lines(focus: FocusArea, theme: &ThemeColors) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    section(&mut lines, "This panel", panel_bindings(focus), theme);
    section(&mut lines, "Universal", UNIVERSAL, theme);
    section(&mut lines, "Global (Ctrl)", GLOBAL, theme);
    section(&mut lines, "While editing chat", EDITING, theme);
    section(&mut lines, "Slash commands", COMMANDS, theme);
    lines
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
        use FocusArea::*;
        for f in [
            ChatInput,
            FileExplorer,
            CodeEditor,
            ModelSelector,
            Settings,
            CodeReview,
            PerformanceDashboard,
            ProviderHealth,
            ProjectAnalyzer,
            GitCommit,
            FeatureNavigator,
            ByteBotPanel,
            CollaborationHub,
            VoiceInterface,
            TerminalAssistant,
            SecurityAuditor,
            PerformanceProfiler,
            CustomModels,
            LearningMode,
            MultiLanguage,
            ReviewDashboard,
            TaskManager,
            WorktreePanel,
            AdvisePanel,
        ] {
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
            "Universal",
            "Global (Ctrl)",
            "While editing chat",
            "Slash commands",
        ] {
            assert!(text.contains(section), "missing {section}");
        }
        assert!(text.contains("Ctrl+Y"));
        assert!(text.contains("scroll chat"));
    }
}
