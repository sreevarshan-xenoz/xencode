//! Whether a program is working, idle, waiting on the person, or finished —
//! read only from the signals a terminal program sends on purpose.

use std::time::{Duration, Instant};

/// No output for this long reads as idle.
pub const IDLE_AFTER: Duration = Duration::from_secs(2);

/// What a tile's frame and rail row say about its program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState {
    Working,
    Idle,
    NeedsYou,
    Done,
    Failed,
}

/// Where the scan is inside an escape sequence, so a bell that ends an OSC
/// (a window title, say) is not taken for a call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Scan {
    #[default]
    Text,
    Escape,
    Osc,
    OscEscape,
}

/// Reads a program's output for attention signals.
#[derive(Debug, Clone)]
pub struct Attention {
    scan: Scan,
    /// The OSC being read, up to its first separator: `9`, `777`, `0`, …
    osc: String,
    last_output: Option<Instant>,
    needs_you: bool,
    started: Instant,
}

impl Default for Attention {
    fn default() -> Self {
        Attention {
            scan: Scan::Text,
            osc: String::new(),
            last_output: None,
            needs_you: false,
            started: Instant::now(),
        }
    }
}

impl Attention {
    /// Take in a chunk of output that arrived at `now`.
    pub fn observe(&mut self, bytes: &[u8], now: Instant) {
        if bytes.is_empty() {
            return;
        }
        self.last_output = Some(now);
        for &b in bytes {
            self.scan = match (self.scan, b) {
                (Scan::Text, 0x07) => {
                    self.needs_you = true;
                    Scan::Text
                }
                (Scan::Text, 0x1b) => Scan::Escape,
                (Scan::Text, _) => Scan::Text,
                (Scan::Escape, b']') => {
                    self.osc.clear();
                    Scan::Osc
                }
                (Scan::Escape, _) => Scan::Text,
                (Scan::Osc, 0x07) => {
                    self.osc_ended();
                    Scan::Text
                }
                (Scan::Osc, 0x1b) => Scan::OscEscape,
                (Scan::Osc, c) => {
                    if self.osc.len() < 8 && c != b';' && !self.osc.contains(';') {
                        self.osc.push(c as char);
                    } else if c == b';' && !self.osc.contains(';') {
                        self.osc.push(';');
                    }
                    Scan::Osc
                }
                // ESC \ ends the OSC; anything else after ESC abandons it.
                (Scan::OscEscape, b'\\') => {
                    self.osc_ended();
                    Scan::Text
                }
                (Scan::OscEscape, _) => Scan::Text,
            };
        }
    }

    fn osc_ended(&mut self) {
        let number = self.osc.split(';').next().unwrap_or_default();
        if number == "9" || number == "777" {
            self.needs_you = true;
        }
        self.osc.clear();
    }

    /// The person typed into the program: whatever it was asking is answered.
    pub fn typed(&mut self) {
        self.needs_you = false;
        self.last_output = Some(Instant::now());
    }

    /// The state at `now`, given the program's exit code if it has ended.
    pub fn state(&self, now: Instant, exited: Option<i32>) -> TileState {
        match exited {
            Some(0) => return TileState::Done,
            Some(_) => return TileState::Failed,
            None => {}
        }
        if self.needs_you {
            return TileState::NeedsYou;
        }
        let since = self.last_output.unwrap_or(self.started);
        if now.saturating_duration_since(since) >= IDLE_AFTER {
            TileState::Idle
        } else {
            TileState::Working
        }
    }
}
