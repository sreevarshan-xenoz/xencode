//! Windows 11 draws a one-pixel border and rounded corners around every
//! window, including a transparent one, so the badge would sit inside a faint
//! square with a shadow. Asking the window manager for no border colour,
//! square corners and no frame drawing leaves only what the badge draws. Other platforms draw no such frame.

use raw_window_handle::HasWindowHandle;

#[cfg(windows)]
pub fn remove_frame(window: &impl HasWindowHandle) {
    use raw_window_handle::RawWindowHandle;
    use windows_sys::Win32::Graphics::Dwm::{
        DwmSetWindowAttribute, DWMNCRP_DISABLED, DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE,
        DWMWA_NCRENDERING_POLICY, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND,
    };
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return;
    };
    let hwnd = win32.hwnd.get() as windows_sys::Win32::Foundation::HWND;
    let no_border: u32 = DWMWA_COLOR_NONE;
    let square = DWMWCP_DONOTROUND;
    // No frame drawing at all: no top edge line and no drop shadow.
    let no_frame = DWMNCRP_DISABLED;
    // SAFETY: `hwnd` is this process's own live window, and each value points
    // at a local of exactly the size passed with it. A failure only leaves the
    // frame visible.
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_BORDER_COLOR as u32,
            (&no_border as *const u32).cast(),
            std::mem::size_of::<u32>() as u32,
        );
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE as u32,
            (&square as *const i32).cast(),
            std::mem::size_of::<i32>() as u32,
        );
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_NCRENDERING_POLICY as u32,
            (&no_frame as *const i32).cast(),
            std::mem::size_of::<i32>() as u32,
        );
    }
}

#[cfg(not(windows))]
pub fn remove_frame(_window: &impl HasWindowHandle) {}
