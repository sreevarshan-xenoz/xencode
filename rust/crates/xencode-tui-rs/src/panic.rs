//! Leaving the terminal usable when the interface falls over.
//!
//! The TUI runs on a raw-mode terminal with the alternate screen and mouse
//! capture switched on. A panic in that state used to skip the restore code
//! entirely: the message was printed into a screen the user could no longer
//! scroll or clear, their shell came back with keypresses echoing wrong, and
//! nothing recorded that it had happened. This module puts a hook in front of
//! the default panic behaviour that does all three of the missing things.

use std::backtrace::Backtrace;
use std::panic::PanicHookInfo;
use std::path::PathBuf;

/// Put the terminal back the way the shell expects, ignoring anything that
/// fails: this runs while the process is already dying, and a second failure
/// here must not stop the rest of the restore.
pub fn restore_terminal() {
    let _ = crossterm::terminal::disable_raw_mode();
    let mut stdout = std::io::stdout();
    let _ = crossterm::execute!(
        stdout,
        crossterm::event::DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    let _ = std::io::Write::flush(&mut stdout);
}

/// The text recorded for one panic: what panicked, where, and the backtrace if
/// `RUST_BACKTRACE` asked for one.
pub fn panic_report(info: &PanicHookInfo) -> String {
    let payload = info.payload();
    let message = if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(non-string panic payload)".to_string()
    };
    let location = info
        .location()
        .map(|l| format!("{}:{}", l.file(), l.line()))
        .unwrap_or_else(|| "unknown location".to_string());
    // `Backtrace::capture` is the standard-library answer to RUST_BACKTRACE:
    // it returns a disabled status, cheaply, unless the environment asked.
    let backtrace = Backtrace::capture();
    let backtrace = match backtrace.status() {
        std::backtrace::BacktraceStatus::Captured => format!("\n{}", backtrace),
        _ => String::new(),
    };
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("time_unix: {secs}\nmessage: {message}\nlocation: {location}{backtrace}\n")
}

/// Install the hook. `record` is where the last panic is written; it is the
/// only place a crash is recoverable after the screen has been restored, and
/// `xencode doctor` reads it.
pub fn install_panic_hook(record: impl Into<PathBuf>) {
    let record = record.into();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let report = panic_report(info);
        // Owner-only: a panic message can quote a file path or a prompt.
        let _ = xencode_core_rs::write_atomic(&record, report.as_bytes());
        eprintln!(
            "xencode crashed; the crash is recorded in {}",
            record.display()
        );
        restore_terminal();
        previous(info);
    }));
}

/// Path used by the CLI: alongside the config, so one directory holds
/// everything a bug report needs.
pub fn default_record_path() -> Option<PathBuf> {
    xencode_config_rs::XencodeConfig::config_dir()
        .ok()
        .map(|dir| dir.join("last_panic.log"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    /// The process-wide panic hook is one resource per test binary, and these
    /// tests are the only writers, so they take turns.
    static HOOK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn record_path() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-panic-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("last_panic.log")
    }

    /// The real thing: install the hook, panic, and see what was left behind.
    #[test]
    fn an_actual_panic_is_recorded_and_the_hook_runs_before_the_default_one() {
        let _guard = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let record = record_path();
        let previous = std::panic::take_hook();
        install_panic_hook(&record);
        let panicked = std::panic::catch_unwind(|| {
            panic!("rendering the file panel gave up");
        })
        .is_err();
        // Put the process-wide hook back so no other test sees this one.
        std::panic::set_hook(previous);

        assert!(panicked, "the test itself must panic to exercise the hook");
        let text = std::fs::read_to_string(&record).unwrap();
        assert!(
            text.contains("rendering the file panel gave up"),
            "record was: {text}"
        );
        assert!(text.contains("location:"), "record was: {text}");
        assert!(
            text.contains("panic.rs"),
            "the location should name the file that panicked: {text}"
        );
        std::fs::remove_dir_all(record.parent().unwrap()).unwrap();
    }

    #[test]
    fn the_record_is_readable_only_by_the_owner() {
        let _guard = HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let record = record_path();
        let previous = std::panic::take_hook();
        install_panic_hook(&record);
        let _ = std::panic::catch_unwind(|| panic!("permissions check"));
        std::panic::set_hook(previous);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&record).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        std::fs::remove_dir_all(record.parent().unwrap()).unwrap();
    }

    #[test]
    fn restoring_a_terminal_that_was_never_changed_is_harmless() {
        // Headless test runs have no alternate screen to leave. This must not
        // panic or abort the process that is already crashing.
        restore_terminal();
    }
}
