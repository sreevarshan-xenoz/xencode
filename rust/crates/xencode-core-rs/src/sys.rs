//! The questions xencode asks the operating system about processes and disks,
//! answered once per platform (`PL-1`).
//!
//! Every crate that needs to know whether a pid is still running, to stop one,
//! or to price a write against free space comes here instead of calling
//! `libc` directly, so a Unix-only call cannot leak into a crate that also
//! ships on Windows. Each answer is a real system call on both platforms; where
//! a platform cannot answer a question the function says so in its return
//! value rather than guessing.

use std::path::Path;

/// True when the process with `pid` exists and has not exited. `0` is never a
/// pid we track, so it is never alive.
///
/// On Linux a zombie (a child that exited but was not reaped yet) counts as
/// dead, because it will never do more work. Elsewhere on Unix the answer is a
/// `kill(pid, 0)` probe, which sends no signal. On Windows the process is
/// opened for a query and alive means its exit code is still `STILL_ACTIVE`.
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    imp::pid_alive(pid)
}

/// Stop a process we started: ask first, force after a short grace period.
/// Calling it on a pid that is already gone does nothing.
///
/// On Unix that is `SIGTERM`, then `SIGKILL` if the process is still alive
/// half a second later. Windows has no polite request that every process
/// honours, so there the process is ended with `TerminateProcess` straight
/// away and the grace period only waits for it to disappear.
pub fn terminate(pid: u32) {
    if !pid_alive(pid) {
        return;
    }
    imp::request_stop(pid);
    for _ in 0..5 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if pid_alive(pid) {
        imp::force_stop(pid);
    }
}

/// Bytes a non-root writer can still put on the filesystem holding `dir`,
/// which must be an existing directory. `None` means the system refused to
/// answer, which is not the same as a full disk.
pub fn free_disk_bytes(dir: &Path) -> Option<u64> {
    imp::free_disk_bytes(dir)
}

#[cfg(unix)]
mod imp {
    use std::path::Path;

    #[cfg(target_os = "linux")]
    pub fn pid_alive(pid: u32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => match stat.rfind(')') {
                Some(i) => stat.as_bytes().get(i + 2) != Some(&b'Z'),
                None => true,
            },
            Err(_) => false,
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn pid_alive(pid: u32) -> bool {
        // SAFETY: signal 0 checks that the pid exists and sends nothing.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }

    pub fn request_stop(pid: u32) {
        // SAFETY: kill(2) on a pid we started; a pid that is already gone only
        // sets errno.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    }

    pub fn force_stop(pid: u32) {
        // SAFETY: as above.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }

    pub fn free_disk_bytes(dir: &Path) -> Option<u64> {
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: `stat` is a fully-sized destination and `c_dir` is a
        // NUL-terminated path that outlives the call.
        if unsafe { libc::statvfs(c_dir.as_ptr(), &mut stat) } != 0 {
            return None;
        }
        // Available blocks, not free blocks: the difference is the space the
        // filesystem keeps for root, which a download cannot use.
        let block = (stat.f_frsize as u64).max(1);
        Some(stat.f_bavail as u64 * block)
    }
}

#[cfg(windows)]
mod imp {
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, TerminateProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        PROCESS_TERMINATE,
    };

    pub fn pid_alive(pid: u32) -> bool {
        // SAFETY: OpenProcess returns null on failure and a handle we own on
        // success; the handle is closed before returning.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return false;
            }
            let mut code: u32 = 0;
            let ok = GetExitCodeProcess(handle, &mut code) != 0;
            CloseHandle(handle);
            ok && code == STILL_ACTIVE as u32
        }
    }

    pub fn request_stop(pid: u32) {
        force_stop(pid);
    }

    pub fn force_stop(pid: u32) {
        // SAFETY: as in `pid_alive`; TerminateProcess on a handle opened with
        // PROCESS_TERMINATE.
        unsafe {
            let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
            if handle.is_null() {
                return;
            }
            TerminateProcess(handle, 1);
            CloseHandle(handle);
        }
    }

    pub fn free_disk_bytes(dir: &Path) -> Option<u64> {
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut available: u64 = 0;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the two
        // totals we do not need may be null.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut available,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        (ok != 0).then_some(available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_process_is_alive_and_pid_zero_is_not() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
    }

    #[test]
    fn a_child_is_alive_until_terminated_and_dead_after() {
        #[cfg(unix)]
        let mut child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
        #[cfg(windows)]
        let mut child = std::process::Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();
        assert!(pid_alive(pid));
        terminate(pid);
        // Reap it so a Linux zombie does not linger past the assertion.
        let _ = child.wait();
        assert!(!pid_alive(pid));
    }

    #[test]
    fn the_temp_directory_reports_some_free_space() {
        let free = free_disk_bytes(&std::env::temp_dir()).expect("the system answers");
        assert!(free > 0);
    }
}
