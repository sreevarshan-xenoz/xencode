//! OM-1 against a real shell in a real pseudo-terminal: `cmd` on Windows, `sh`
//! elsewhere. Nothing here is simulated — the bytes a test reads are the bytes
//! the shell wrote.

use std::time::{Duration, Instant};

use ratatui::{buffer::Buffer, layout::Rect};
use xencode_term_rs::{Attention, CommandSpec, Screen, Session, Size, TileState};

fn shell() -> CommandSpec {
    let cwd = std::env::temp_dir();
    if cfg!(windows) {
        CommandSpec {
            program: "cmd".into(),
            args: vec!["/Q".into()],
            cwd,
            env: vec![],
        }
    } else {
        CommandSpec {
            program: "sh".into(),
            args: vec![],
            cwd,
            env: vec![("PS1".into(), "$ ".into())],
        }
    }
}

/// Enter, as the shell on this platform reads it from a terminal.
fn line(text: &str) -> Vec<u8> {
    format!("{text}\r").into_bytes()
}

/// Feed `screen` from `session` until `needle` shows or ten seconds pass.
fn wait_for(session: &Session, screen: &mut Screen, needle: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        session.pump(screen);
        if screen.text().contains(needle) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn wait_exit(session: &Session) -> Option<i32> {
    // The shell's questions are still answered while it is waited for.
    let mut screen = Screen::new(Size { cols: 80, rows: 24 });
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Some(code) = session.exit_status() {
            return Some(code);
        }
        // Keep the pipe drained so a chatty child cannot block on a full one.
        session.pump(&mut screen);
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

#[test]
fn output_reaches_the_screen_and_input_reaches_the_program() {
    let size = Size { cols: 80, rows: 24 };
    let session = Session::spawn(&shell(), size).unwrap();
    let mut screen = Screen::new(size);
    // What is typed is not what is printed: the shell has to run the line to
    // turn it into the word, so the word on screen is the program's output and
    // not the terminal's echo of the keys.
    let typed = if cfg!(windows) {
        "echo XEN%COMSPEC:~0,0%CODE_TERM"
    } else {
        "echo XEN''CODE_TERM"
    };
    session.write(&line(typed)).unwrap();
    assert!(
        wait_for(&session, &mut screen, "XENCODE_TERM"),
        "{}",
        screen.text()
    );
    session.kill();
}

#[test]
fn a_child_that_exits_reports_its_code() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    session.write(&line("exit 3")).unwrap();
    assert_eq!(wait_exit(&session), Some(3));
}

#[test]
fn resize_changes_the_size_the_program_sees() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    let mut screen = Screen::new(Size { cols: 80, rows: 24 });
    session
        .resize(Size {
            cols: 101,
            rows: 33,
        })
        .unwrap();
    screen.resize(Size {
        cols: 101,
        rows: 33,
    });
    if cfg!(windows) {
        session.write(&line("mode con")).unwrap();
        assert!(wait_for(&session, &mut screen, "101"), "{}", screen.text());
    } else {
        session.write(&line("stty size")).unwrap();
        assert!(
            wait_for(&session, &mut screen, "33 101"),
            "{}",
            screen.text()
        );
    }
    session.kill();
}

#[test]
fn stop_ends_the_program() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    // A command that would run for minutes if nothing ended it.
    let long = if cfg!(windows) {
        "ping -n 300 127.0.0.1"
    } else {
        "sleep 300"
    };
    session.write(&line(long)).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    session.kill();
    assert!(wait_exit(&session).is_some(), "the shell kept running");
}

/// On Unix the whole process group goes, so a child the shell started in the
/// background does not outlive the tile.
#[cfg(unix)]
#[test]
fn stop_ends_what_the_program_started_too() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    let mut screen = Screen::new(Size { cols: 80, rows: 24 });
    session.write(&line("sleep 300 & echo PID=$!")).unwrap();
    // The typed line itself shows `PID=$!`; the pid is the `PID=` followed
    // by digits that the shell printed.
    let printed = |text: &str| {
        text.split("PID=")
            .skip(1)
            .filter_map(|rest| rest.split_whitespace().next()?.parse::<u32>().ok())
            .last()
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while printed(&screen.text()).is_none() && Instant::now() < deadline {
        session.pump(&mut screen);
        std::thread::sleep(Duration::from_millis(50));
    }
    let pid = printed(&screen.text()).expect("the background job's pid");
    session.kill();
    wait_exit(&session);
    let deadline = Instant::now() + Duration::from_secs(5);
    while std::path::Path::new(&format!("/proc/{pid}")).exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !std::path::Path::new(&format!("/proc/{pid}")).exists(),
        "the background child {pid} outlived the tile"
    );
}

#[test]
fn a_bell_from_a_real_program_raises_needs_you() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    let mut screen = Screen::new(Size { cols: 80, rows: 24 });
    let mut attention = Attention::default();
    // The ESC-free way each shell can be told to ring the bell.
    let ring = if cfg!(windows) {
        "powershell -NoProfile -Command \"[char]7\""
    } else {
        "printf '\\a'"
    };
    session.write(&line(ring)).unwrap();
    attention.typed();
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        let bytes = session.pump(&mut screen);
        attention.observe(&bytes, Instant::now());
        if attention.state(Instant::now(), None) == TileState::NeedsYou {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(attention.state(Instant::now(), None), TileState::NeedsYou);
    session.kill();
}

#[test]
fn attention_follows_the_programs_own_signals() {
    let mut a = Attention::default();
    let t = Instant::now();
    a.observe(b"done\x07", t);
    assert_eq!(a.state(t, None), TileState::NeedsYou);
    a.typed();
    assert_eq!(a.state(t, None), TileState::Working);
    a.observe(b"\x1b]9;finished\x07", t);
    assert_eq!(a.state(t, None), TileState::NeedsYou, "OSC 9");
    a.typed();
    a.observe(b"\x1b]777;notify;title;body\x1b\\", t);
    assert_eq!(a.state(t, None), TileState::NeedsYou, "OSC 777");
    // A bell that ends an OSC title is the title's terminator, not a call.
    let mut b = Attention::default();
    b.observe(b"\x1b]0;my title\x07", t);
    assert_eq!(b.state(t, None), TileState::Working);
    assert_eq!(b.state(t + Duration::from_secs(3), None), TileState::Idle);
    assert_eq!(b.state(t, Some(0)), TileState::Done);
    assert_eq!(b.state(t, Some(2)), TileState::Failed);
}

#[test]
fn replay_reproduces_the_screen() {
    let size = Size { cols: 40, rows: 10 };
    let mut first = Screen::new(size);
    first.feed(b"\x1b[31mred\x1b[0m and plain\r\nsecond line");
    let mut second = Screen::new(size);
    second.feed(&first.replay());
    assert_eq!(first.text(), second.text());
    assert_eq!(
        first.vt().cell(0, 0).map(|c| c.fgcolor()),
        second.vt().cell(0, 0).map(|c| c.fgcolor())
    );
}

#[test]
fn the_screen_draws_into_a_buffer_and_stops_at_the_edge() {
    let mut s = Screen::new(Size { cols: 20, rows: 3 });
    s.feed("wide 漢字 cells and a long line".as_bytes());
    let area = Rect::new(0, 0, 10, 3);
    let mut buf = Buffer::empty(Rect::new(0, 0, 12, 3));
    xencode_term_rs::draw(s.vt(), area, &mut buf);
    // A wide character takes its cell and the one after it, which ratatui
    // leaves blank.
    let row: String = (0..5).map(|x| buf[(x, 0)].symbol().to_string()).collect();
    assert_eq!(row, "wide ");
    assert_eq!(buf[(5, 0)].symbol(), "漢");
    assert_eq!(buf[(7, 0)].symbol(), "字");
    assert_eq!(buf[(10, 0)].symbol(), " ", "nothing past the tile");
    assert_eq!(buf[(11, 0)].symbol(), " ", "nothing past the tile");
}

#[test]
fn a_tile_smaller_than_the_screen_is_clipped_not_a_panic() {
    let mut s = Screen::new(Size { cols: 80, rows: 24 });
    s.feed(b"hello");
    for (w, h) in [(0u16, 0u16), (1, 1), (3, 2), (79, 23)] {
        let mut buf = Buffer::empty(Rect::new(0, 0, w.max(1), h.max(1)));
        xencode_term_rs::draw(s.vt(), Rect::new(0, 0, w, h), &mut buf);
    }
}

#[test]
fn a_flood_of_output_keeps_only_the_scrollback() {
    let mut s = Screen::new(Size { cols: 20, rows: 5 });
    let mut flood = Vec::new();
    for i in 0..20_000 {
        flood.extend_from_slice(format!("line {i}\r\n").as_bytes());
    }
    s.feed(&flood);
    assert!(s.text().contains("line 19999"), "{}", s.text());
    s.scroll(1_000_000);
    assert!(s.scrollback() <= xencode_term_rs::SCROLLBACK);
}
