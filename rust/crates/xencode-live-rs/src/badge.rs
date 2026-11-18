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
    /// The newest graded eval run in this project, filled in by the badge from
    /// the project's own eval log. `None` when there is no graded result.
    pub eval_label: Option<String>,
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
            eval_label: None,
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

/// The newest graded run in a `task_eval.jsonl` log, written for a hover card:
/// `eval 7/8 (88%)`. Lines that do not parse (including a torn final write),
/// rows without a timestamp or grade counts, and runs where nothing reached a
/// verdict are all skipped. `None` means there is no graded result to show —
/// never "0%".
pub fn latest_eval_label(jsonl_text: &str) -> Option<String> {
    let mut newest: Option<(u64, u64, u64)> = None;
    for line in jsonl_text.lines() {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(ts), Some(graded), Some(passed)) = (
            row.get("ts_unix_ms").and_then(serde_json::Value::as_u64),
            row.get("graded").and_then(serde_json::Value::as_u64),
            row.get("passed").and_then(serde_json::Value::as_u64),
        ) else {
            continue;
        };
        // A pass count above the grade count is not a measurement.
        if graded == 0 || passed > graded {
            continue;
        }
        if newest.is_none_or(|(seen, _, _)| ts > seen) {
            newest = Some((ts, graded, passed));
        }
    }
    let (_, graded, passed) = newest?;
    let percent = passed as f64 * 100.0 / graded as f64;
    Some(format!("eval {passed}/{graded} ({percent:.0}%)"))
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
    fn eval_label_picks_the_newest_graded_run_past_a_broken_tail() {
        // A real log is appended oldest first; the last write may be torn, and
        // the most recent completed run may have graded nothing at all.
        let log = [
            r#"{"ts_unix_ms":1000,"graded":8,"passed":5,"model":"m"}"#,
            r#"{"ts_unix_ms":2000,"graded":8,"passed":7,"model":"m"}"#,
            r#"{"ts_unix_ms":3000,"graded":0,"passed":0,"model":"m"}"#,
            r#"{"ts_unix_ms":4000,"graded":8,"passed":8,"model":"m""#,
        ]
        .join("\n");
        assert_eq!(latest_eval_label(&log).as_deref(), Some("eval 7/8 (88%)"));
    }

    #[test]
    fn eval_label_uses_the_largest_timestamp_and_skips_junk_rows() {
        let log = [
            r#"{"ts_unix_ms":9000,"graded":10,"passed":10}"#,
            "not json at all",
            r#"{"ts_unix_ms":5000,"graded":4,"passed":3}"#,
            r#"{"ts_unix_ms":9500,"graded":4,"passed":9}"#,
            r#"{"ts_unix_ms":1000,"graded":4}"#,
            r#"{"ts_unix_ms":9500,"graded":"8","passed":"7"}"#,
        ]
        .join("\n");
        assert_eq!(
            latest_eval_label(&log).as_deref(),
            Some("eval 10/10 (100%)")
        );
    }

    #[test]
    fn eval_label_is_absent_when_no_run_graded_anything() {
        assert_eq!(latest_eval_label(""), None);
        assert_eq!(latest_eval_label("torn \n"), None);
        assert_eq!(
            latest_eval_label(
                "{\"ts_unix_ms\":1,\"graded\":0,\"passed\":0}\n\
                 {\"ts_unix_ms\":2,\"graded\":0,\"passed\":0}\n"
            ),
            None
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
