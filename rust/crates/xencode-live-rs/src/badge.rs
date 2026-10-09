//! What the floating badge shows (DK-2), decided without a window so it can be
//! tested anywhere.

use crate::{LiveState, Read};

/// Seconds without a heartbeat before a session counts as not responding.
pub const STALE_AFTER: u64 = 20;
/// Seconds without a heartbeat before a session is left out entirely.
pub const FORGET_AFTER: u64 = 600;
/// Seconds a finished session keeps the badge green.
pub const FINISHED_FADES_AFTER: u64 = 60;

/// What the resting badge looks like, least to most urgent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shown {
    Nothing,
    Finished,
    Working,
    Failed,
    NeedsYou,
}

/// One session in the hover card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub session_id: String,
    pub project: String,
    pub model: String,
    /// The state in words; the card never relies on colour alone.
    pub state_words: String,
    pub headline: String,
    pub since: String,
    pub responding: bool,
    /// What this session alone would make the badge show.
    pub shown: Shown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadgeView {
    pub shown: Shown,
    /// At least one session stopped sending heartbeats.
    pub stale_dot: bool,
    pub rows: Vec<Row>,
    /// Files that could not be used, as "file: reason".
    pub skipped: Vec<String>,
}

fn rank(shown: Shown) -> u8 {
    match shown {
        Shown::Nothing => 0,
        Shown::Finished => 1,
        Shown::Working => 2,
        Shown::Failed => 3,
        Shown::NeedsYou => 4,
    }
}

fn most_urgent(a: Shown, b: Shown) -> Shown {
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

pub fn view(reads: &[Read], now: u64) -> BadgeView {
    let mut v = BadgeView {
        shown: Shown::Nothing,
        stale_dot: false,
        rows: Vec::new(),
        skipped: Vec::new(),
    };
    for r in reads {
        let s = match r {
            Read::Ok(s) => s,
            Read::Skipped { file, reason } => {
                v.skipped.push(format!("{}: {reason}", file.display()));
                continue;
            }
        };
        let silent_for = now.saturating_sub(s.heartbeat_at);
        if silent_for > FORGET_AFTER {
            continue;
        }
        let responding = silent_for <= STALE_AFTER;
        let age = now.saturating_sub(s.changed_at);
        let shown = if !responding {
            v.stale_dot = true;
            Shown::Nothing
        } else {
            match s.state {
                LiveState::Idle => Shown::Nothing,
                LiveState::Working => Shown::Working,
                LiveState::NeedsYou => Shown::NeedsYou,
                LiveState::Failed => Shown::Failed,
                LiveState::Finished if age <= FINISHED_FADES_AFTER => Shown::Finished,
                LiveState::Finished => Shown::Nothing,
            }
        };
        v.shown = most_urgent(v.shown, shown);
        let state_words = if !responding {
            "not responding"
        } else {
            match s.state {
                LiveState::Idle => "idle",
                LiveState::Working => "working",
                LiveState::NeedsYou => "needs you",
                LiveState::Finished => "finished",
                LiveState::Failed => "failed",
            }
        };
        v.rows.push(Row {
            session_id: s.session_id.clone(),
            project: s.project.clone(),
            model: s.model.clone(),
            state_words: state_words.to_string(),
            headline: s.headline.clone(),
            since: since(age),
            responding,
            shown,
        });
    }
    v
}

impl BadgeView {
    /// The badge once the person has hovered over the failed sessions named in
    /// `seen`: a failure they have read no longer colours the badge red.
    pub fn shown_after(&self, seen: &[String]) -> Shown {
        self.rows
            .iter()
            .map(|row| {
                if row.shown == Shown::Failed && seen.contains(&row.session_id) {
                    Shown::Nothing
                } else {
                    row.shown
                }
            })
            .fold(Shown::Nothing, most_urgent)
    }
}

/// How long ago, in the largest two units.
pub fn since(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m {}s ago", secs / 60, secs % 60),
        _ => format!("{}h {}m ago", secs / 3600, (secs % 3600) / 60),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiveSource, LiveState, LiveStatus, Read, VERSION};

    fn s(id: &str, state: LiveState, changed: u64, beat: u64) -> Read {
        Read::Ok(LiveStatus {
            version: VERSION,
            session_id: id.into(),
            pid: 1,
            project: format!("E:/{id}"),
            model: "m".into(),
            source: LiveSource::Chat,
            state,
            headline: format!("{id} doing"),
            changed_at: changed,
            heartbeat_at: beat,
        })
    }

    #[test]
    fn the_most_urgent_state_wins() {
        let now = 1000;
        let reads = [
            s("a", LiveState::Working, 990, 999),
            s("b", LiveState::NeedsYou, 990, 999),
            s("c", LiveState::Failed, 990, 999),
        ];
        assert_eq!(view(&reads, now).shown, Shown::NeedsYou);
        let reads = [
            s("a", LiveState::Working, 990, 999),
            s("c", LiveState::Failed, 990, 999),
        ];
        assert_eq!(view(&reads, now).shown, Shown::Failed);
    }

    #[test]
    fn two_sessions_are_two_rows() {
        let v = view(
            &[
                s("a", LiveState::Working, 990, 999),
                s("b", LiveState::Idle, 990, 999),
            ],
            1000,
        );
        assert_eq!(v.rows.len(), 2);
        assert_eq!(v.rows[0].state_words, "working");
        assert_eq!(v.rows[1].state_words, "idle");
        assert_eq!(v.rows[0].session_id, "a");
    }

    #[test]
    fn finished_fades_after_a_minute_but_its_row_stays() {
        let v = view(
            &[s(
                "a",
                LiveState::Finished,
                1000 - FINISHED_FADES_AFTER - 1,
                999,
            )],
            1000,
        );
        assert_eq!(v.shown, Shown::Nothing);
        assert_eq!(v.rows[0].state_words, "finished");
        let v = view(&[s("a", LiveState::Finished, 990, 999)], 1000);
        assert_eq!(v.shown, Shown::Finished);
    }

    #[test]
    fn a_silent_session_is_not_responding_then_forgotten() {
        let v = view(
            &[s("a", LiveState::Working, 900, 1000 - STALE_AFTER - 1)],
            1000,
        );
        assert!(v.stale_dot);
        assert!(!v.rows[0].responding);
        assert_eq!(v.rows[0].state_words, "not responding");
        assert_eq!(v.shown, Shown::Nothing, "a dead session is not 'working'");
        let v = view(
            &[s("a", LiveState::Working, 0, 1000 - FORGET_AFTER - 1)],
            1000,
        );
        assert!(v.rows.is_empty());
    }

    #[test]
    fn skipped_files_are_reported_in_words() {
        let reads = [Read::Skipped {
            file: "x.json".into(),
            reason: "written by a newer xencode (format 2); update this badge".into(),
        }];
        let v = view(&reads, 1000);
        assert_eq!(v.shown, Shown::Nothing);
        assert_eq!(
            v.skipped,
            vec!["x.json: written by a newer xencode (format 2); update this badge".to_string()]
        );
    }

    #[test]
    fn since_is_written_in_hours_minutes_and_seconds() {
        assert_eq!(since(65), "1m 5s ago");
        assert_eq!(since(5), "5s ago");
        assert_eq!(since(7300), "2h 1m ago");
    }

    #[test]
    fn a_failure_the_person_has_seen_no_longer_colours_the_badge() {
        let reads = [
            s("a", LiveState::Failed, 990, 999),
            s("b", LiveState::Working, 990, 999),
        ];
        let v = view(&reads, 1000);
        assert_eq!(v.shown_after(&["a".to_string()]), Shown::Working);
        assert_eq!(v.shown_after(&[]), Shown::Failed);
    }
}
