//! The conversation transcript (TD-1): what each message looks like on the
//! conversation screen. Each speaker's turn starts with a coloured `▍` and its
//! name and keeps that marker down its wrapped lines; a tool call is one line,
//! `⎿ <what it did> <how it ended>`, with long output folded.
//!
//! Tool calls travel as ordinary messages with role [`TOOL_ROLE`] and a JSON
//! [`ToolEntry`] body, so everything that already carries `messages` (the
//! engine's view, replay, the editor integration) carries them too.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde::{Deserialize, Serialize};

use crate::app::UiMessage;
use crate::theme::ThemeColors;

/// The role of a message that records one tool call.
pub const TOOL_ROLE: &str = "tool";

/// One tool call and how it ended.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct ToolEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// What the call did in a few words, as the approval prompt names it.
    #[serde(default)]
    pub summary: String,
    /// `ok`, `failed`, `denied`, `refused`, … once it ended.
    #[serde(default)]
    pub outcome: Option<String>,
    /// The start of its result.
    #[serde(default)]
    pub preview: String,
}

/// A message recording `entry`.
pub fn tool_message(entry: &ToolEntry) -> UiMessage {
    UiMessage {
        role: TOOL_ROLE.to_string(),
        content: serde_json::to_string(entry).unwrap_or_default(),
    }
}

/// The tool call `msg` records, when it records one.
pub fn read_tool(msg: &UiMessage) -> Option<ToolEntry> {
    if msg.role != TOOL_ROLE {
        return None;
    }
    serde_json::from_str(&msg.content).ok()
}

/// How the transcript is laid out.
#[derive(Clone, Debug, Default)]
pub struct RenderOpts {
    /// Columns available, including the marker.
    pub width: u16,
    /// Show every tool call's whole output.
    pub expand_tools: bool,
    /// The one tool call whose output is shown in full.
    pub expanded: Option<String>,
}

/// Lines of a tool call's output shown before it folds.
const TOOL_PREVIEW_LINES: usize = 4;

/// The transcript's lines for `messages`.
pub fn lines(messages: &[UiMessage], theme: &ThemeColors, opts: &RenderOpts) -> Vec<Line<'static>> {
    let width = (opts.width as usize).max(8);
    let mut out = Vec::new();
    for msg in messages {
        match msg.role.as_str() {
            "user" => speaker(
                &mut out,
                "You",
                theme.message_user,
                plain_lines(&msg.content, theme.fg),
                width,
            ),
            "assistant" => speaker(
                &mut out,
                "xencode",
                theme.accent,
                crate::markdown::render_markdown(&msg.content, theme),
                width,
            ),
            "system" => {
                for row in wrap_indented(&msg.content, width, 4) {
                    let rest = row.get(4..).unwrap_or("").to_string();
                    out.push(Line::from(vec![
                        Span::styled("  · ", Style::default().fg(theme.message_system)),
                        Span::styled(rest, Style::default().fg(theme.message_system)),
                    ]));
                }
            }
            TOOL_ROLE => match read_tool(msg) {
                Some(entry) => tool_lines(&mut out, &entry, theme, opts, width),
                None => dim(&mut out, &msg.content, theme, width),
            },
            _ => dim(&mut out, &msg.content, theme, width),
        }
    }
    out
}

fn dim(out: &mut Vec<Line<'static>>, text: &str, theme: &ThemeColors, width: usize) {
    for row in wrap_indented(text, width, 2) {
        out.push(Line::from(Span::styled(
            row,
            Style::default().fg(theme.message_system),
        )));
    }
}

fn plain_lines(text: &str, fg: ratatui::style::Color) -> Vec<Line<'static>> {
    clean(text)
        .split('\n')
        .map(|l| Line::from(Span::styled(l.to_string(), Style::default().fg(fg))))
        .collect()
}

/// A speaker's turn: a blank line, `▍Name`, then `body` wrapped with `▍ `
/// down the left edge.
fn speaker(
    out: &mut Vec<Line<'static>>,
    name: &str,
    color: ratatui::style::Color,
    body: Vec<Line<'static>>,
    width: usize,
) {
    if !out.is_empty() {
        out.push(Line::from(""));
    }
    out.push(Line::from(vec![
        Span::styled("▍", Style::default().fg(color)),
        Span::styled(
            name.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]));
    let marker = vec![Span::styled("▍ ", Style::default().fg(color))];
    for line in body {
        out.extend(wrap_spans(line.spans, width, &marker));
    }
}

/// One tool call: `  ⎿ summary  ✓`, then its output, folded unless asked.
fn tool_lines(
    out: &mut Vec<Line<'static>>,
    entry: &ToolEntry,
    theme: &ThemeColors,
    opts: &RenderOpts,
    width: usize,
) {
    let what = if entry.summary.trim().is_empty() {
        entry.name.clone()
    } else {
        entry.summary.clone()
    };
    let (glyph, color) = match entry.outcome.as_deref() {
        None => ("…", theme.info),
        Some("ok" | "done" | "finished") => ("✓", theme.success),
        Some(_) => ("✕", theme.danger),
    };
    let ended = match entry.outcome.as_deref() {
        Some(o) if !matches!(o, "ok" | "done" | "finished") => format!(" {o}"),
        _ => String::new(),
    };
    let tail = 2 + 1 + ended.chars().count();
    let room = width.saturating_sub(tail);
    let mut head = format!("  ⎿ {}", clean(&what).replace('\n', " "));
    if unicode_width::UnicodeWidthStr::width(head.as_str()) > room {
        head = head
            .chars()
            .take(room.saturating_sub(1))
            .collect::<String>()
            + "…";
    }
    out.push(Line::from(vec![
        Span::styled(head, Style::default().fg(theme.message_system)),
        Span::raw("  "),
        Span::styled(glyph, Style::default().fg(color)),
        Span::styled(ended, Style::default().fg(color)),
    ]));
    let preview: Vec<String> = wrap_indented(entry.preview.trim_end(), width, 4)
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect();
    let all = opts.expand_tools || opts.expanded.as_deref() == Some(entry.id.as_str());
    let shown = if all {
        preview.len()
    } else {
        preview.len().min(TOOL_PREVIEW_LINES)
    };
    for l in &preview[..shown] {
        out.push(Line::from(Span::styled(
            l.clone(),
            Style::default().fg(theme.message_system),
        )));
    }
    if shown < preview.len() {
        out.push(Line::from(Span::styled(
            format!("    +{} lines · ctrl+o", preview.len() - shown),
            Style::default()
                .fg(theme.message_system)
                .add_modifier(Modifier::ITALIC),
        )));
    }
}

/// Text with terminal escapes and control characters removed and tabs
/// expanded, so a command's output cannot restyle the screen.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => match chars.peek() {
                // CSI: ESC [ … final byte in @..~
                Some('[') => {
                    chars.next();
                    for d in chars.by_ref() {
                        if ('@'..='~').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … BEL or ESC \
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\u{7}' {
                            break;
                        }
                        if d == '\u{1b}' {
                            chars.next();
                            break;
                        }
                    }
                }
                _ => {
                    chars.next();
                }
            },
            '\t' => out.push_str("    "),
            '\n' => out.push('\n'),
            '\r' => {}
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// `text` as plain lines of at most `width` columns, each starting with
/// `indent` spaces.
pub fn wrap_indented(text: &str, width: usize, indent: usize) -> Vec<String> {
    let prefix = " ".repeat(indent);
    let cleaned = clean(text);
    let mut out = Vec::new();
    for raw in cleaned.split('\n') {
        let spans = vec![Span::raw(raw.to_string())];
        for line in wrap_spans(spans, width, &[Span::raw(prefix.clone())]) {
            out.push(line.spans.iter().map(|s| s.content.as_ref()).collect());
        }
    }
    out
}

/// One character and the style it is drawn in.
type Cell = (char, Style);

/// `spans` wrapped at word boundaries to `width` columns, each line starting
/// with `prefix`; a word wider than a line is cut. An empty input still gives
/// one line (the prefix alone).
fn wrap_spans(
    spans: Vec<Span<'static>>,
    width: usize,
    prefix: &[Span<'static>],
) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthChar;
    let prefix_width: usize = prefix
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    let room = width.saturating_sub(prefix_width).max(1);
    let cw = |c: char| c.width().unwrap_or(0);
    // Runs of word characters or of spaces, each with its style.
    let mut runs: Vec<(bool, Vec<Cell>)> = Vec::new();
    for span in &spans {
        for c in span.content.chars() {
            let space = c == ' ';
            match runs.last_mut() {
                Some((was_space, run)) if *was_space == space => run.push((c, span.style)),
                _ => runs.push((space, vec![(c, span.style)])),
            }
        }
    }
    let mut lines: Vec<Vec<Cell>> = vec![Vec::new()];
    let mut used = 0usize;
    for (space, run) in runs {
        let run_width: usize = run.iter().map(|(c, _)| cw(*c)).sum();
        if space {
            if used == 0 && lines.len() > 1 {
                // Spaces where a line was broken are dropped.
                continue;
            }
            if used + run_width <= room {
                used += run_width;
                lines.last_mut().unwrap().extend(run);
            } else {
                lines.push(Vec::new());
                used = 0;
            }
            continue;
        }
        // A word that fits on a fresh line but not on this one moves down.
        if used > 0 && used + run_width > room && run_width <= room {
            lines.push(Vec::new());
            used = 0;
        }
        for cell in run {
            let w = cw(cell.0);
            if used + w > room && used > 0 {
                lines.push(Vec::new());
                used = 0;
            }
            used += w;
            lines.last_mut().unwrap().push(cell);
        }
    }
    lines
        .into_iter()
        .map(|mut cells| {
            while cells.last().is_some_and(|(c, _)| *c == ' ') {
                cells.pop();
            }
            let mut out: Vec<Span<'static>> = prefix.to_vec();
            let mut current = String::new();
            let mut style: Option<Style> = None;
            for (c, s) in cells {
                if style != Some(s) {
                    if let Some(prev) = style {
                        out.push(Span::styled(std::mem::take(&mut current), prev));
                    }
                    style = Some(s);
                }
                current.push(c);
            }
            if let Some(prev) = style {
                out.push(Span::styled(current, prev));
            }
            Line::from(out)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    fn text(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    fn opts(width: u16) -> RenderOpts {
        RenderOpts {
            width,
            expand_tools: false,
            expanded: None,
        }
    }

    #[test]
    fn wrapped_lines_keep_their_indent() {
        let out = wrap_indented("one two three four five six", 12, 2);
        assert!(out.len() > 1, "{out:?}");
        assert!(out.iter().all(|l| l.starts_with("  ")), "{out:?}");
        assert!(out.iter().all(|l| l.width() <= 12), "{out:?}");
    }

    /// Review focus 1: a word wider than the line is cut inside the indent.
    #[test]
    fn a_word_wider_than_the_line_is_cut_inside_the_indent() {
        let url = "https://example.invalid/".to_string() + &"a".repeat(80);
        let out = wrap_indented(&url, 20, 2);
        assert!(out.len() >= 4, "{out:?}");
        assert!(
            out.iter().all(|l| l.starts_with("  ") && l.width() <= 20),
            "{out:?}"
        );
    }

    /// Review focus 2: tool output is plain text.
    #[test]
    fn tool_output_is_plain_text() {
        let out = wrap_indented("a\tb\u{1b}[31mred\u{1b}[0m\u{7}", 40, 0);
        assert_eq!(out, vec!["a    bred".to_string()]);
    }

    #[test]
    fn each_speaker_gets_a_marker_and_a_name_down_its_lines() {
        let msgs = vec![
            UiMessage {
                role: "user".into(),
                content: "fix the failing parser test now please".into(),
            },
            UiMessage {
                role: "assistant".into(),
                content: "done".into(),
            },
        ];
        let t = text(&lines(&msgs, &ThemeColors::get("ocean"), &opts(20)));
        assert!(t.iter().any(|l| l.starts_with("▍You")), "{t:?}");
        assert!(t.iter().any(|l| l.starts_with("▍xencode")), "{t:?}");
        let body: Vec<&String> = t
            .iter()
            .filter(|l| l.contains("parser") || l.contains("now"))
            .collect();
        assert!(
            !body.is_empty() && body.iter().all(|l| l.starts_with("▍ ")),
            "{t:?}"
        );
        assert!(t.iter().all(|l| l.width() <= 20), "{t:?}");
    }

    #[test]
    fn a_tool_call_is_one_line_and_long_output_folds() {
        let entry = ToolEntry {
            id: "c1".into(),
            name: "run_command".into(),
            arguments: serde_json::json!({"command": "cargo test"}),
            summary: "cargo test".into(),
            outcome: Some("failed".into()),
            preview: (1..=20)
                .map(|n| format!("line {n}"))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        let msgs = vec![tool_message(&entry)];
        assert_eq!(read_tool(&msgs[0]), Some(entry.clone()));
        let theme = ThemeColors::get("ocean");
        let folded = text(&lines(&msgs, &theme, &opts(60)));
        assert!(
            folded
                .iter()
                .any(|l| l.contains("⎿") && l.contains("cargo test")),
            "{folded:?}"
        );
        assert!(
            folded
                .iter()
                .any(|l| l.contains("+16 lines") && l.contains("ctrl+o")),
            "{folded:?}"
        );
        assert!(!folded.iter().any(|l| l.contains("line 20")));
        let open = text(&lines(
            &msgs,
            &theme,
            &RenderOpts {
                width: 60,
                expand_tools: true,
                expanded: None,
            },
        ));
        assert!(open.iter().any(|l| l.contains("line 20")), "{open:?}");
        let one = text(&lines(
            &msgs,
            &theme,
            &RenderOpts {
                width: 60,
                expand_tools: false,
                expanded: Some("c1".into()),
            },
        ));
        assert!(one.iter().any(|l| l.contains("line 20")), "{one:?}");
    }

    /// Review focus 5: an older window or replay has no tool role; an
    /// unknown role is still shown.
    #[test]
    fn an_unknown_role_is_a_plain_line() {
        let msgs = vec![UiMessage {
            role: "observer".into(),
            content: "hello".into(),
        }];
        let t = text(&lines(&msgs, &ThemeColors::get("ocean"), &opts(40)));
        assert!(t.iter().any(|l| l.contains("hello")), "{t:?}");
        let bad = UiMessage {
            role: TOOL_ROLE.into(),
            content: "not json".into(),
        };
        assert_eq!(read_tool(&bad), None);
        let t = text(&lines(&[bad], &ThemeColors::get("ocean"), &opts(40)));
        assert!(t.iter().any(|l| l.contains("not json")), "{t:?}");
    }
}
