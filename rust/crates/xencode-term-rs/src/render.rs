//! A `vt100` screen drawn into a ratatui buffer, cell by cell.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
};

fn color(c: vt100::Color) -> Color {
    match c {
        vt100::Color::Default => Color::Reset,
        vt100::Color::Idx(i) => Color::Indexed(i),
        vt100::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
    }
}

/// Draw `screen` into `area` of `buf`: each cell's text, colours, bold,
/// italic, underline and reverse. A wide character takes its two columns; a
/// screen larger than `area` is clipped at its right and bottom edges, and
/// nothing is written outside `area` or `buf`.
pub fn draw(screen: &vt100::Screen, area: Rect, buf: &mut Buffer) {
    let area = area.intersection(buf.area);
    if area.width == 0 || area.height == 0 {
        return;
    }
    let (rows, cols) = screen.size();
    for row in 0..rows.min(area.height) {
        for col in 0..cols.min(area.width) {
            let Some(cell) = screen.cell(row, col) else {
                continue;
            };
            if cell.is_wide_continuation() {
                continue;
            }
            // A wide character whose second half would fall past the edge is
            // left out rather than drawn half.
            if cell.is_wide() && col + 1 >= area.width {
                continue;
            }
            let mut modifier = Modifier::empty();
            if cell.bold() {
                modifier |= Modifier::BOLD;
            }
            if cell.italic() {
                modifier |= Modifier::ITALIC;
            }
            if cell.underline() {
                modifier |= Modifier::UNDERLINED;
            }
            if cell.inverse() {
                modifier |= Modifier::REVERSED;
            }
            let style = Style::default()
                .fg(color(cell.fgcolor()))
                .bg(color(cell.bgcolor()))
                .add_modifier(modifier);
            let text = cell.contents();
            let symbol = if text.is_empty() { " " } else { text.as_str() };
            let target = &mut buf[(area.x + col, area.y + row)];
            target.set_symbol(symbol);
            target.set_style(style);
        }
    }
}
