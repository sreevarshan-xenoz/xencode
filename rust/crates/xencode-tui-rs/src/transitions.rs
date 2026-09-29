//! Why the window arrangement is the way it is (`V-9`).
//!
//! The question worth answering is not "reproduce last Tuesday's screen" —
//! `GL-3` settled that the project resumes by re-verifying and never by
//! replaying. It is "**why is this pane here**", asked while the screen is
//! still in front of you. So every change to the arrangement records the ask
//! that caused it, in the words a keystroke deserves, and one panel lists
//! them for the session.
//!
//! Three rules hold this together:
//!
//! - **Session memory, not a fourth file.** The log lives in `App` and dies
//!   with it. `layout.json` stores where the panes were (`V-6`) and
//!   `config.json` stores what they should look like (`V-5`, `V-4`); a log of
//!   why would be a new format to promise forever, for a list that is only
//!   interesting next to the screen it describes.
//! - **One kind of row.** Nothing that did not change the arrangement is
//!   recorded here. A toast or a warning is not a transition, and a log that
//!   mixes the two cannot answer the question it exists for.
//! - **Dedup by result, not by chance.** A repeated keystroke that lands the
//!   screen somewhere new stays — five presses of a resize chord are five
//!   rows — while a keystroke that lands it where the previous row already
//!   says it is, adds nothing to know twice. Rows are compared by
//!   [`Transition::signature`] rather than by the line a reader is shown,
//!   because that line carries pane sizes the terminal dictates as well as the
//!   arrangement the user changed.
//!
//! Two kinds of thing are deliberately not rows. An overlay — the agent stack
//! `Ctrl+N` puts over the body, a permission prompt — is not the arrangement:
//! it covers the screen for a moment and leaves it, and a log of the body that
//! also listed every popup would stop answering the question it was built for.
//! And a change with no keystroke in it — a pane opened because a worker's
//! state changed — has no variant yet, because nothing opens a pane that way.
//! `V-10` is the item that does, and it records the state it observed as its
//! trigger's words.

/// The ask behind one arrangement change. Renders as a complete phrase: the
/// panel reads `trigger → arrangement`, and a bare word would leave the
/// reader supplying the verb.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// What the screen was when the session opened. `restored` says whether an
    /// arrangement came back from `layout.json` (`V-6`) or the configured
    /// layout rendered fresh — the difference a reader asking "why is this pane
    /// here" on the first frame actually wants, and a fact the app knows rather
    /// than one a description has to go and read off the disk to guess.
    SessionOpened { restored: bool },
    /// `Alt+Left` / `Alt+Right` (`V-3`). `words` is the chord as pressed and
    /// the window it moved; `grew` is which way the focused pane went.
    ResizeChord { words: String, grew: bool },
    /// `Ctrl+U` (`V-5`), which names the layout it landed on.
    LayoutCycle { name: String },
    /// `Ctrl+1`…`Ctrl+9` (`V-4`), which put a saved view on screen.
    ViewRecalled { name: String, slot: usize },
    /// `Ctrl+Shift+<digit>` (`V-4`), which stored the screen — and renames the
    /// arrangement on the header, which is a change to what is on it.
    ViewStored { name: String, slot: usize },
    /// A divider dragged by hand (`V-7`). `divider` names the two panes the
    /// line separates, `cells` is how far the pointer went and `points` how
    /// much of it the clamps allowed. The pair can be un-nameable — a template
    /// whose side resolves to no pane a reader would name — and the row says
    /// so rather than inventing one.
    DividerDrag {
        divider: Option<String>,
        cells: i16,
        points: i16,
    },
    /// `Ctrl+T`, which puts the terminal strip in the chat column or takes it
    /// out. The strip's absence below its minimum height is the renderer's
    /// clamp, not a second row here.
    TerminalStrip { shown: bool },
}

impl Trigger {
    pub(crate) fn words(&self) -> String {
        match self {
            Trigger::SessionOpened { restored } => {
                if *restored {
                    "session opened on the arrangement saved from last time".to_string()
                } else {
                    "session opened on the configured layout".to_string()
                }
            }
            Trigger::ResizeChord { words, grew } => format!(
                "{words} {} the focused pane",
                if *grew { "grew" } else { "shrank" }
            ),
            Trigger::LayoutCycle { name } => format!("Ctrl+U cycled to {name}"),
            Trigger::ViewRecalled { name, slot } => {
                format!("Ctrl+{slot} recalled the view {name}")
            }
            Trigger::ViewStored { name, slot } => {
                format!("Ctrl+Shift+{slot} stored this screen as {name}")
            }
            Trigger::DividerDrag {
                divider,
                cells,
                points,
            } => match divider {
                Some(pair) => format!("dragged the {pair} divider {cells} cells, took {points}"),
                None => format!("dragged a divider {cells} cells, took {points}"),
            },
            Trigger::TerminalStrip { shown } => format!(
                "Ctrl+T put the terminal strip {} the screen",
                if *shown { "on" } else { "off" }
            ),
        }
    }
}

/// One arrangement change: what asked for it, and where the screen ended up.
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    /// Seconds after the session opened, so the list reads in the order it
    /// happened without a wall clock that would make two runs disagree.
    pub at: f64,
    pub trigger: Trigger,
    /// The arrangement as it was before. A row's `before` is the row above
    /// its `after`, because the log is a chain kept live — which is what lets
    /// the detail view show a change without walking the list for it.
    pub before: String,
    /// The arrangement as it is now, in one line: name, then each pane with the
    /// box it was drawn into, then the focused one — `classic · Files 23x40,
    /// Code 62x40, Chat 30x22, Input 30x17 → Chat`. For reading, in the detail
    /// view, next to the row's trigger.
    pub after: String,
    /// The same arrangement in a form for comparing, not reading: the rows are
    /// deduped by this. The drawn boxes cannot be the comparison, because they
    /// follow the terminal's size as much as they follow the arrangement — a
    /// window dragged wider by the window manager would look, on the next
    /// chord, like a change this list never recorded. What is in here instead is
    /// the tree's own encoding, whose shares only move when the arrangement
    /// does.
    pub signature: String,
}

/// Rows kept for the session. The list is bounded so a long-lived pane cannot
/// grow it forever; the oldest rows go first, which is the half a reader is
/// least likely to still want.
pub const TRANSITION_LIMIT: usize = 200;

/// Append `entry`, dropping the oldest rows past [`TRANSITION_LIMIT`], and
/// report whether it was kept — a change that left the screen exactly where
/// the last row already said it was is not a thing worth reading twice. The
/// comparison is `entry.signature` against `previous`, the signature of the
/// row before it, or `None` for the session's opening row, which has no
/// previous and is always kept.
pub fn record(log: &mut Vec<Transition>, entry: Transition, previous: Option<&str>) -> bool {
    if Some(entry.signature.as_str()) == previous {
        return false;
    }
    log.push(entry);
    if log.len() > TRANSITION_LIMIT {
        let excess = log.len() - TRANSITION_LIMIT;
        log.drain(..excess);
    }
    true
}

/// A row's elapsed time as `m:ss`. Long enough a session to reach an hour
/// still fits the column; a log that cannot be scanned is not worth reading.
pub fn elapsed(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(after: &str, trigger: Trigger) -> Transition {
        entry_with(after, after, trigger)
    }

    /// The display line and the comparison key are two different strings, so a
    /// test that wants them apart says both.
    fn entry_with(after: &str, signature: &str, trigger: Trigger) -> Transition {
        Transition {
            at: 0.0,
            trigger,
            before: "the screen before".to_string(),
            after: after.to_string(),
            signature: signature.to_string(),
        }
    }

    fn chord(grew: bool) -> Trigger {
        Trigger::ResizeChord {
            words: "Alt+Right".to_string(),
            grew,
        }
    }

    #[test]
    fn a_change_to_a_new_arrangement_is_kept() {
        let mut log = Vec::new();
        assert!(record(&mut log, entry("code → Editor", chord(true)), None));
        assert_eq!(log.len(), 1);
    }

    /// A keystroke that lands the screen where it already was — a resize the
    /// clamps refuse at a pane's minimum, say — would otherwise fill the list
    /// with rows that describe no change at all.
    #[test]
    fn a_keystroke_that_changed_nothing_is_not_a_row() {
        let mut log = Vec::new();
        record(&mut log, entry("code → Editor", chord(true)), None);
        assert!(
            !record(
                &mut log,
                entry("code → Editor", chord(false)),
                Some("code → Editor"),
            ),
            "the same arrangement was recorded twice"
        );
        assert_eq!(log.len(), 1);
    }

    /// A window dragged wider by the window manager changes every pane's box
    /// and no part of the arrangement. The next keystroke after it must not
    /// look like a change the log had never seen, which is why the comparison
    /// ignores the sizes a reader is shown.
    #[test]
    fn sizes_the_terminal_chose_are_not_a_change_in_the_arrangement() {
        let mut log = Vec::new();
        record(
            &mut log,
            entry_with("Files 20x40 → Chat", "files chat", chord(true)),
            None,
        );
        assert!(
            !record(
                &mut log,
                entry_with("Files 20x50 → Chat", "files chat", chord(true)),
                Some("files chat"),
            ),
            "same tree, taller window"
        );
        assert_eq!(log.len(), 1);
    }

    /// The first row is the screen the session opened on. It is kept because
    /// nothing preceded it — which is what makes a list that starts with a
    /// restored arrangement readable rather than a lone trigger with no
    /// arrangement to compare it against.
    #[test]
    fn the_opening_row_is_kept_because_nothing_came_before_it() {
        let mut log = Vec::new();
        assert!(record(
            &mut log,
            entry("classic", Trigger::SessionOpened { restored: true }),
            None
        ));
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn the_log_stays_bounded_and_keeps_the_recent_end() {
        let mut log = Vec::new();
        for i in 0..(TRANSITION_LIMIT + 25) {
            let before = format!("row {i}");
            record(
                &mut log,
                entry(&format!("arrangement {i}"), chord(i % 2 == 0)),
                Some(&before),
            );
        }
        assert_eq!(log.len(), TRANSITION_LIMIT);
        assert_eq!(
            log.last().unwrap().after,
            format!("arrangement {}", TRANSITION_LIMIT + 24)
        );
        assert_eq!(log.first().unwrap().after, "arrangement 25");
    }

    #[test]
    fn a_trigger_says_what_was_asked_in_a_whole_phrase() {
        assert_eq!(
            Trigger::LayoutCycle {
                name: "zen".to_string()
            }
            .words(),
            "Ctrl+U cycled to zen"
        );
        assert_eq!(
            Trigger::DividerDrag {
                divider: Some("Code / Chat".to_string()),
                cells: 6,
                points: 5,
            }
            .words(),
            "dragged the Code / Chat divider 6 cells, took 5"
        );
        assert_eq!(
            Trigger::ViewRecalled {
                name: "Review".to_string(),
                slot: 5,
            }
            .words(),
            "Ctrl+5 recalled the view Review"
        );
        // A drag that travelled further than the clamps allowed says both
        // numbers, because the difference between them is the thing the user
        // felt as the line stopping.
        assert_eq!(
            Trigger::DividerDrag {
                divider: Some("Files / Code".to_string()),
                cells: 40,
                points: 0,
            }
            .words(),
            "dragged the Files / Code divider 40 cells, took 0"
        );
    }

    #[test]
    fn elapsed_time_is_minutes_and_seconds() {
        assert_eq!(elapsed(0.0), "0:00");
        assert_eq!(elapsed(9.4), "0:09");
        assert_eq!(elapsed(65.0), "1:05");
        assert_eq!(elapsed(3729.0), "62:09");
    }
}
