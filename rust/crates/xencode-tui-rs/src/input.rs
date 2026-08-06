//! Shared cursor-aware text buffer for panels with text input fields.
//!
//! Consolidates the repeated `String + cursor` pairs scattered across `App`
//! (commit_message/commit_cursor, bytebot_command/bytebot_cursor,
//! terminal_input/terminal_cursor, settings_url_buffer/settings_url_cursor).
//! Introduced in Step 0; used by panels in later tasks.

/// A text buffer with a cursor position.
#[derive(Clone, Default)]
pub struct TextInput {
    pub text: String,
    pub cursor: usize,
}

impl TextInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }

    pub fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.text.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn cursor_right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor += 1;
        }
    }

    pub fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    pub fn cursor_end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn set(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.cursor = s.len();
        self.text = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_backspace() {
        let mut t = TextInput::new();
        t.insert_char('h');
        t.insert_char('i');
        assert_eq!(t.text, "hi");
        assert_eq!(t.cursor, 2);
        t.backspace();
        assert_eq!(t.text, "h");
        assert_eq!(t.cursor, 1);
    }

    #[test]
    fn test_cursor_movement() {
        let mut t = TextInput::new();
        t.set("abc");
        assert_eq!(t.cursor, 3);
        t.cursor_left();
        t.cursor_left();
        assert_eq!(t.cursor, 1);
        t.insert_char('X');
        assert_eq!(t.text, "aXbc");
    }

    #[test]
    fn test_is_blank() {
        let mut t = TextInput::new();
        assert!(t.is_blank());
        t.set("   ");
        assert!(t.is_blank());
        t.set(" x ");
        assert!(!t.is_blank());
    }

    #[test]
    fn test_delete_at_cursor() {
        let mut t = TextInput::new();
        t.set("abc");
        t.cursor_left(); // cursor at 2
        t.delete(); // removes 'c'
        assert_eq!(t.text, "ab");
        assert_eq!(t.cursor, 2);
    }
}
