//! What a program drew: a `vt100` screen with its scrollback, and the
//! answers a terminal owes the program for the questions it asks.

use crate::session::Size;

/// Rows of history a tile keeps above what is on screen.
pub const SCROLLBACK: usize = 5_000;

/// The questions a terminal answers, as the program sends them.
const QUERIES: [&[u8]; 4] = [b"\x1b[6n", b"\x1b[5n", b"\x1b[c", b"\x1b[0c"];

/// A program's screen, fed from its output.
pub struct Screen {
    parser: vt100::Parser,
    /// The end of the last chunk when it could be the start of a query split
    /// across two reads.
    pending: Vec<u8>,
    /// Answers owed to the program, oldest first.
    replies: Vec<u8>,
}

impl Screen {
    pub fn new(size: Size) -> Self {
        Screen {
            parser: vt100::Parser::new(size.rows.max(1), size.cols.max(1), SCROLLBACK),
            pending: Vec::new(),
            replies: Vec::new(),
        }
    }

    /// Take in what the program wrote. A cursor-position, status or
    /// device-attributes query is answered from this screen at the point it
    /// was asked (Windows' ConPTY asks where the cursor is before it passes
    /// any output on, and waits for the answer); `take_replies` hands the
    /// answers over.
    pub fn feed(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut data = std::mem::take(&mut self.pending);
        data.extend_from_slice(bytes);
        let mut start = 0;
        let mut at = 0;
        while at < data.len() {
            if data[at] != 0x1b {
                at += 1;
                continue;
            }
            let rest = &data[at..];
            if let Some(query) = QUERIES.iter().find(|q| rest.starts_with(q)) {
                let end = at + query.len();
                self.parser.process(&data[start..end]);
                self.answer(query);
                start = end;
                at = end;
                continue;
            }
            // The chunk may end part way through a query: keep that tail for
            // the next read instead of drawing it.
            if QUERIES.iter().any(|q| q.starts_with(rest)) {
                self.parser.process(&data[start..at]);
                self.pending = rest.to_vec();
                return;
            }
            at += 1;
        }
        self.parser.process(&data[start..]);
    }

    fn answer(&mut self, query: &[u8]) {
        match query {
            b"\x1b[6n" => {
                let (row, col) = self.parser.screen().cursor_position();
                self.replies
                    .extend_from_slice(format!("\x1b[{};{}R", row + 1, col + 1).as_bytes());
            }
            b"\x1b[5n" => self.replies.extend_from_slice(b"\x1b[0n"),
            // A VT100 with advanced video, the answer most programs expect.
            _ => self.replies.extend_from_slice(b"\x1b[?1;2c"),
        }
    }

    /// The answers owed to the program since the last call, to be written to
    /// it.
    pub fn take_replies(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.replies)
    }

    pub fn resize(&mut self, size: Size) {
        self.parser.set_size(size.rows.max(1), size.cols.max(1));
    }

    /// The bytes that redraw this screen on a fresh one: what a window that
    /// starts watching a running program is sent first.
    pub fn replay(&self) -> Vec<u8> {
        self.parser.screen().contents_formatted()
    }

    /// Look `rows` further back (positive) or forward (negative) in the
    /// scrollback; it stops at either end.
    pub fn scroll(&mut self, rows: isize) {
        let now = self.parser.screen().scrollback() as isize;
        let to = (now + rows).max(0) as usize;
        self.parser.set_scrollback(to.min(SCROLLBACK));
    }

    /// How far back the view is, in rows; 0 follows the program.
    pub fn scrollback(&self) -> usize {
        self.parser.screen().scrollback()
    }

    /// The screen as plain text, one line per row.
    pub fn text(&self) -> String {
        self.parser.screen().contents()
    }

    /// The underlying screen, for drawing.
    pub fn vt(&self) -> &vt100::Screen {
        self.parser.screen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cursor_query_is_answered_where_the_cursor_is() {
        let mut s = Screen::new(Size { cols: 20, rows: 5 });
        s.feed(b"ab\r\ncde\x1b[6n");
        assert_eq!(s.take_replies(), b"\x1b[2;4R");
        assert!(s.take_replies().is_empty());
    }

    #[test]
    fn a_query_split_across_two_reads_is_still_answered() {
        let mut s = Screen::new(Size { cols: 20, rows: 5 });
        s.feed(b"x\x1b[");
        assert!(s.take_replies().is_empty());
        s.feed(b"6ny");
        assert_eq!(s.take_replies(), b"\x1b[1;2R");
        assert!(s.text().starts_with("xy"), "{}", s.text());
    }

    #[test]
    fn status_and_device_queries_get_the_standard_answers() {
        let mut s = Screen::new(Size { cols: 20, rows: 5 });
        s.feed(b"\x1b[5n\x1b[c");
        assert_eq!(s.take_replies(), b"\x1b[0n\x1b[?1;2c");
    }
}
