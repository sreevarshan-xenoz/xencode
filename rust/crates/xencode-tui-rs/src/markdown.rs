//! Minimal markdown → ratatui `Line` renderer for chat messages (E3-01).
//!
//! No new dependencies. Block level: fenced code (including an unterminated
//! fence while streaming), headings, list items, block quotes, rules.
//! Inline: `code`, **bold**, *italic*. Everything else passes through as text.

use crate::theme::ThemeColors;
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

pub fn render_markdown(content: &str, theme: &ThemeColors) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_fence = false;

    for raw in content.lines() {
        if raw.trim_start().starts_with("```") {
            if in_fence {
                in_fence = false;
                lines.push(Line::from(Span::styled(
                    "  └─",
                    Style::default().fg(theme.message_system),
                )));
            } else {
                in_fence = true;
                let lang = raw.trim_start().trim_start_matches('`').trim();
                let label = if lang.is_empty() {
                    "  ┌─ code".to_string()
                } else {
                    format!("  ┌─ {lang}")
                };
                lines.push(Line::from(Span::styled(
                    label,
                    Style::default().fg(theme.message_system),
                )));
            }
            continue;
        }

        if in_fence {
            lines.push(Line::from(Span::styled(
                format!("    {raw}"),
                Style::default().fg(theme.message_assistant),
            )));
            continue;
        }

        let trimmed = raw.trim_start();
        if trimmed.is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        if let Some((marker, body)) = heading(trimmed) {
            let mut spans = vec![Span::styled(
                format!("  {marker} "),
                Style::default().fg(theme.accent).add_modifier(Modifier::BOLD),
            )];
            spans.extend(inline_spans(body, theme, Modifier::BOLD));
            lines.push(Line::from(spans));
            continue;
        }

        if matches!(trimmed, "---" | "***" | "___") {
            lines.push(Line::from(Span::styled(
                "  ────────────────────────────",
                Style::default().fg(theme.border),
            )));
            continue;
        }

        if let Some(body) = trimmed.strip_prefix("> ") {
            lines.push(Line::from(Span::styled(
                format!("  │ {body}"),
                Style::default()
                    .fg(theme.message_system)
                    .add_modifier(Modifier::ITALIC),
            )));
            continue;
        }

        let mut spans = Vec::new();
        if let Some((marker, body)) = list_item(trimmed) {
            spans.push(Span::styled(
                format!("  {marker} "),
                Style::default().fg(theme.accent),
            ));
            spans.extend(inline_spans(body, theme, Modifier::empty()));
        } else {
            spans.push(Span::raw("  "));
            spans.extend(inline_spans(trimmed, theme, Modifier::empty()));
        }
        lines.push(Line::from(spans));
    }

    if in_fence {
        // Unterminated fence = mid-stream; still close the visual box.
        lines.push(Line::from(Span::styled(
            "  └─",
            Style::default().fg(theme.message_system),
        )));
    }
    lines
}

/// ATX heading: returns ("###", text) for `### text` (levels 1-6).
fn heading(trimmed: &str) -> Option<(String, &str)> {
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if rest.is_empty() || rest.starts_with(' ') {
        Some(("#".repeat(level), rest.trim_start()))
    } else {
        None
    }
}

/// "- " / "* " / "+ " bullets and "1. " style ordered items.
fn list_item(trimmed: &str) -> Option<(String, &str)> {
    for b in ["- ", "* ", "+ "] {
        if let Some(body) = trimmed.strip_prefix(b) {
            return Some(("•".to_string(), body));
        }
    }
    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let rest = trimmed.strip_prefix(&digits)?;
    let body = rest
        .strip_prefix(". ")
        .or_else(|| if rest == "." { Some("") } else { None })?;
    Some((format!("{digits}."), body))
}

/// Split one line into spans on `code`, **bold**, *italic* markers.
/// Unclosed markers are emitted as literal text (streaming-friendly).
fn inline_spans(text: &str, theme: &ThemeColors, base: Modifier) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '`' => {
                if let Some(end) = find_char(&chars, i + 1, '`') {
                    push_plain(&mut spans, &mut plain, theme, base, Modifier::empty());
                    let code: String = chars[i + 1..end].iter().collect();
                    spans.push(Span::styled(
                        code,
                        Style::default()
                            .fg(theme.highlight)
                            .add_modifier(Modifier::BOLD),
                    ));
                    i = end + 1;
                } else {
                    plain.push('`');
                    i += 1;
                }
            }
            '*' if chars.get(i + 1) == Some(&'*') => {
                if let Some(end) = find_pair(&chars, i + 2, '*') {
                    push_plain(&mut spans, &mut plain, theme, base, Modifier::empty());
                    let bold: String = chars[i + 2..end].iter().collect();
                    spans.push(Span::styled(
                        bold,
                        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                    ));
                    i = end + 2;
                } else {
                    plain.push('*');
                    plain.push('*');
                    i += 2;
                }
            }
            '*' => {
                if let Some(end) = find_char(&chars, i + 1, '*') {
                    push_plain(&mut spans, &mut plain, theme, base, Modifier::empty());
                    let italic: String = chars[i + 1..end].iter().collect();
                    spans.push(Span::styled(
                        italic,
                        Style::default()
                            .fg(theme.fg)
                            .add_modifier(Modifier::ITALIC),
                    ));
                    i = end + 1;
                } else {
                    plain.push('*');
                    i += 1;
                }
            }
            c => {
                plain.push(c);
                i += 1;
            }
        }
    }
    push_plain(&mut spans, &mut plain, theme, base, Modifier::empty());
    spans
}

fn push_plain(
    spans: &mut Vec<Span<'static>>,
    plain: &mut String,
    theme: &ThemeColors,
    base: Modifier,
    extra: Modifier,
) {
    if !plain.is_empty() {
        let mut style = Style::default().fg(theme.fg);
        for m in [base, extra] {
            if !m.is_empty() {
                style = style.add_modifier(m);
            }
        }
        spans.push(Span::styled(std::mem::take(plain), style));
    }
}

fn find_char(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == target)
}

fn find_pair(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len().saturating_sub(1)).find(|&j| chars[j] == target && chars[j + 1] == target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> ThemeColors {
        ThemeColors::get("ocean")
    }

    fn plain_text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn fenced_block_is_marked_and_indented() {
        let out = render_markdown("before\n```rust\nfn x() {}\n```\nafter", &theme());
        let text = plain_text(&out);
        assert!(text.contains("┌─ rust"), "{text}");
        assert!(text.contains("    fn x() {}"), "{text}");
        assert!(text.contains("└─"), "{text}");
        assert!(text.starts_with("  before"));
        assert!(text.ends_with("  after"));
        // Code line carries its own style, not plain fg.
        let code_line = out.iter().find(|l| l.spans[0].content.contains("fn x")).unwrap();
        assert_eq!(code_line.spans[0].style.fg, Some(theme().message_assistant));
    }

    #[test]
    fn unterminated_fence_while_streaming_still_closes() {
        let out = render_markdown("```py\nprint(1)", &theme());
        let text = plain_text(&out);
        assert!(text.contains("┌─ py"));
        assert!(text.contains("    print(1)"));
        assert!(text.contains("└─"), "streamed fence must not leave open box: {text}");
    }

    #[test]
    fn headings_lists_quotes_render_markers() {
        let out = render_markdown("# Title\n- item\n1. step\n> quoted", &theme());
        let text = plain_text(&out);
        assert!(text.contains("# Title"), "{text}");
        assert!(text.contains("• item"), "{text}");
        assert!(text.contains("1. step"), "{text}");
        assert!(text.contains("│ quoted"), "{text}");
    }

    #[test]
    fn inline_code_bold_italic_split_into_spans() {
        let out = render_markdown("use `cargo test`, it is **green** and *fast*", &theme());
        assert_eq!(out.len(), 1);
        let kinds: Vec<(String, Style)> = out[0]
            .spans
            .iter()
            .map(|s| (s.content.to_string(), s.style))
            .collect();
        let code = kinds.iter().find(|(c, _)| c == "cargo test").unwrap();
        assert_eq!(code.1.fg, Some(theme().highlight));
        let bold = kinds.iter().find(|(c, _)| c == "green").unwrap();
        assert!(bold.1.add_modifier.contains(Modifier::BOLD));
        let italic = kinds.iter().find(|(c, _)| c == "fast").unwrap();
        assert!(italic.1.add_modifier.contains(Modifier::ITALIC));
        // No leftover markers anywhere.
        assert!(!kinds.iter().any(|(c, _)| c.contains('`') || c.contains('*')));
    }

    #[test]
    fn unclosed_inline_markers_stay_literal() {
        let out = render_markdown("a `b c", &theme());
        assert_eq!(plain_text(&out), "  a `b c");
        let out2 = render_markdown("still *streaming", &theme());
        assert_eq!(plain_text(&out2), "  still *streaming");
    }

    #[test]
    fn plain_text_gets_indent_only() {
        let out = render_markdown("hello\nworld", &theme());
        assert_eq!(plain_text(&out), "  hello\n  world");
    }
}
