//! One program in a pseudo-terminal: spawn, read, write, resize, stop.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};

/// What to run, where, and with which extra environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    /// Added to the person's own environment, never replacing it: a tile runs
    /// the agent as if they had typed its name in their own terminal.
    pub env: Vec<(String, String)>,
}

/// A terminal size in cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub cols: u16,
    pub rows: u16,
}

impl Size {
    fn pty(self) -> PtySize {
        PtySize {
            rows: self.rows.max(1),
            cols: self.cols.max(1),
            pixel_width: 0,
            pixel_height: 0,
        }
    }
}

/// Why a session could not do what it was asked, in words a tile can show.
#[derive(Debug)]
pub struct TermError(pub String);

impl std::fmt::Display for TermError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TermError {}

fn err(what: &str, e: impl std::fmt::Display) -> TermError {
    TermError(format!("{what}: {e}"))
}

/// A running program and its pseudo-terminal.
pub struct Session {
    master: Mutex<Box<dyn MasterPty + Send>>,
    writer: Mutex<Box<dyn Write + Send>>,
    child: Mutex<Box<dyn Child + Send + Sync>>,
    /// Bytes the reader thread has taken from the terminal and nobody has
    /// asked for yet.
    output: Arc<Mutex<Vec<u8>>>,
    pid: Option<u32>,
    exit: Mutex<Option<i32>>,
}

impl Session {
    /// Start `cmd` in a new pseudo-terminal of `size`.
    pub fn spawn(cmd: &CommandSpec, size: Size) -> Result<Session, TermError> {
        let pair = native_pty_system()
            .openpty(size.pty())
            .map_err(|e| err("could not open a pseudo-terminal", e))?;
        let mut builder = CommandBuilder::new(&cmd.program);
        builder.args(&cmd.args);
        builder.cwd(&cmd.cwd);
        for (key, value) in &cmd.env {
            builder.env(key, value);
        }
        let child = pair
            .slave
            .spawn_command(builder)
            .map_err(|e| err(&format!("could not start {}", cmd.program), e))?;
        // The slave end belongs to the child now; holding it here would keep
        // the terminal open after the child exits, so the reader never sees
        // the end of its output.
        drop(pair.slave);
        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| err("could not read the terminal", e))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| err("could not write to the terminal", e))?;
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        std::thread::Builder::new()
            .name(format!("term-read-{}", cmd.program))
            .spawn(move || {
                let mut chunk = [0u8; 8192];
                loop {
                    match reader.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if let Ok(mut out) = sink.lock() {
                                out.extend_from_slice(&chunk[..n]);
                            }
                        }
                    }
                }
            })
            .map_err(|e| err("could not start the terminal reader", e))?;
        let pid = child.process_id();
        Ok(Session {
            master: Mutex::new(pair.master),
            writer: Mutex::new(writer),
            child: Mutex::new(child),
            output,
            pid,
            exit: Mutex::new(None),
        })
    }

    /// Send `bytes` to the program, as if typed.
    pub fn write(&self, bytes: &[u8]) -> Result<(), TermError> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| TermError("the terminal writer is unavailable".into()))?;
        writer
            .write_all(bytes)
            .and_then(|_| writer.flush())
            .map_err(|e| err("could not write to the program", e))
    }

    /// Tell the program its terminal is now `size`.
    pub fn resize(&self, size: Size) -> Result<(), TermError> {
        self.master
            .lock()
            .map_err(|_| TermError("the terminal is unavailable".into()))?
            .resize(size.pty())
            .map_err(|e| err("could not resize the terminal", e))
    }

    /// Everything the program wrote since the last call.
    pub fn try_read(&self) -> Vec<u8> {
        self.output
            .lock()
            .map(|mut out| std::mem::take(&mut *out))
            .unwrap_or_default()
    }

    /// Read what the program wrote into `screen`, write back the answers the
    /// screen owes it (a terminal has to answer the program's questions, or
    /// some programs, and Windows' ConPTY itself, wait for ever), and return
    /// the bytes read, for anything else that watches them.
    pub fn pump(&self, screen: &mut crate::Screen) -> Vec<u8> {
        let bytes = self.try_read();
        screen.feed(&bytes);
        let replies = screen.take_replies();
        if !replies.is_empty() {
            let _ = self.write(&replies);
        }
        bytes
    }

    /// The program's exit code once it has ended.
    pub fn exit_status(&self) -> Option<i32> {
        let mut exit = self.exit.lock().ok()?;
        if exit.is_none() {
            if let Ok(mut child) = self.child.lock() {
                if let Ok(Some(status)) = child.try_wait() {
                    *exit = Some(status.exit_code() as i32);
                }
            }
        }
        *exit
    }

    /// The program's process id, while the system knows it.
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }

    /// End the program and everything it started: on Unix its whole process
    /// group (the child leads a new session), on Windows its process tree.
    pub fn kill(&self) {
        if let Some(pid) = self.pid {
            kill_tree(pid);
        }
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.exit_status().is_none() {
            self.kill();
        }
    }
}

#[cfg(unix)]
fn kill_tree(pid: u32) {
    let Some(group) = group_target(pid) else {
        return;
    };
    // The child was started as a session leader, so its session id is its
    // pid. An interactive shell puts each background job in a process group
    // of its own, so the whole session is ended, not only the child's group:
    // a job the agent left running in the background goes with the tile.
    let _ = std::process::Command::new("pkill")
        .args(["-KILL", "-s", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    // And the group, where `pkill` is not installed. This is the system call,
    // not the `kill` program: procps-ng's `kill -KILL -164` reads `-164` as
    // options and signals -1, which is every process the user owns. On CI
    // that killed the runner itself.
    // SAFETY: kill(2) takes no pointers; `group` is a negative pid below -1.
    unsafe {
        libc::kill(group, libc::SIGKILL);
    }
}

/// The process group to signal for a session led by `pid`, as the negative
/// number kill(2) expects. None for 0 and 1: as targets, 0 is the caller's
/// own group and -1 is every process the user may signal, and `pkill -s 0`
/// is pkill's own session.
#[cfg(unix)]
fn group_target(pid: u32) -> Option<libc::pid_t> {
    let pid = libc::pid_t::try_from(pid).ok()?;
    (pid > 1).then_some(-pid)
}

#[cfg(windows)]
fn kill_tree(pid: u32) {
    let _ = std::process::Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(all(test, unix))]
mod tests {
    use super::group_target;

    /// As kill(2) targets, 0 is the caller's own group and -1 is every process
    /// the user owns, so neither is ever signalled; a real leader's group is.
    #[test]
    fn only_a_real_session_leader_is_ever_signalled() {
        assert_eq!(group_target(0), None);
        assert_eq!(group_target(1), None);
        assert_eq!(group_target(164), Some(-164));
        assert_eq!(group_target(u32::MAX), None);
    }
}
