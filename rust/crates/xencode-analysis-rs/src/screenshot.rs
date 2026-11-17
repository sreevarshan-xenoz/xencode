//! One-action screen capture — turn what is on the screen into an image file
//! that the existing image pipeline can attach.
//!
//! This module does exactly one thing: shell out to a capture program that is
//! really installed, write a PNG into the system temp directory, verify the
//! bytes, and hand the path back. It deliberately knows nothing about models,
//! prompts or base64 — [`crate::images::prepare_for_send`] and
//! [`crate::images::to_data_url`] already do that part, and callers reuse them.
//!
//! Tool selection is explicit. On Wayland `grim` is used; on X11 `import` (from
//! ImageMagick) then `scrot`; on macOS `screencapture`; on Windows a PowerShell
//! `.NET` one-liner. The `XENCODE_SCREENSHOT_TOOL` environment override names a
//! program on `PATH` (or `none` to refuse), which is useful for unusual setups
//! and for testing the no-tool path without a fake process.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Name of an optional environment variable that picks the capture program.
/// `none`, `off` or `disabled` refuse capture outright; any other value is
/// looked up on `PATH` and used instead of the platform default.
pub const SCREENSHOT_TOOL_ENV: &str = "XENCODE_SCREENSHOT_TOOL";

/// The first eight bytes of a PNG file.
pub const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, thiserror::Error)]
pub enum ScreenshotError {
    /// No capture program could be found or used. The message says what to
    /// install, the same way the voice recorder reports a missing recorder.
    #[error("{0}")]
    NoTool(String),
    /// A capture program was found but exited unsuccessfully.
    #[error("screen capture with {tool} failed: {detail}")]
    CaptureFailed { tool: String, detail: String },
    /// A capture program reported success but did not leave a PNG behind.
    #[error("screen capture with {tool} wrote no PNG at {path}")]
    NoPng { tool: String, path: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Grim,
    Import,
    Scrot,
    Screencapture,
    Powershell,
}

impl Tool {
    fn name(self) -> &'static str {
        match self {
            Tool::Grim => "grim",
            Tool::Import => "import",
            Tool::Scrot => "scrot",
            Tool::Screencapture => "screencapture",
            Tool::Powershell => "powershell",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "grim" => Some(Tool::Grim),
            "import" => Some(Tool::Import),
            "scrot" => Some(Tool::Scrot),
            "screencapture" => Some(Tool::Screencapture),
            "powershell" | "pwsh" => Some(Tool::Powershell),
            _ => None,
        }
    }

    /// Command line that tells this program to write one PNG to `path`.
    /// Arguments are passed as a vector — no shell string is ever built.
    fn args(self, path: &Path) -> Vec<String> {
        let p = path.to_string_lossy().into_owned();
        match self {
            Tool::Grim => vec![p],
            Tool::Import => vec!["-window".to_string(), "root".to_string(), p],
            Tool::Scrot => vec![p],
            Tool::Screencapture => vec!["-x".to_string(), p],
            Tool::Powershell => vec![
                "-NoProfile".to_string(),
                "-Command".to_string(),
                powershell_script(&p),
            ],
        }
    }
}

/// A PowerShell command that copies the virtual screen into a PNG. The path is
/// quoted for PowerShell (a single quote inside it becomes two single quotes).
fn powershell_script(path: &str) -> String {
    let quoted = path.replace('\'', "''");
    format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         Add-Type -AssemblyName System.Drawing; \
         $bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen; \
         $image = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height); \
         $graphics = [System.Drawing.Graphics]::FromImage($image); \
         $graphics.CopyFromScreen($bounds.Left, $bounds.Top, 0, 0, $image.Size); \
         $image.Save('{quoted}', [System.Drawing.Imaging.ImageFormat]::Png); \
         $graphics.Dispose(); $image.Dispose()"
    )
}

/// Programs to try, in order. An explicit override is honoured on every
/// platform so the same code can be exercised and repaired by a user; with no
/// override the choice follows the display server actually present.
fn tools_for(override_value: Option<&str>) -> Vec<Tool> {
    match override_value.map(str::trim).filter(|s| !s.is_empty()) {
        Some("none" | "off" | "disabled") => Vec::new(),
        Some(name) => Tool::from_name(name).into_iter().collect(),
        None => default_tools(),
    }
}

fn default_tools() -> Vec<Tool> {
    #[cfg(target_os = "linux")]
    {
        let wayland = std::env::var("WAYLAND_DISPLAY")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        let x11 = std::env::var("DISPLAY")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        if wayland {
            vec![Tool::Grim, Tool::Import, Tool::Scrot]
        } else if x11 {
            vec![Tool::Import, Tool::Scrot]
        } else {
            // No display advertised. Still try the programs so a setup with a
            // nonstandard environment gets a real error from the tool rather
            // than a guess from us.
            vec![Tool::Grim, Tool::Import, Tool::Scrot]
        }
    }
    #[cfg(target_os = "macos")]
    {
        vec![Tool::Screencapture]
    }
    #[cfg(target_os = "windows")]
    {
        vec![Tool::Powershell]
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Vec::new()
    }
}

fn no_tool_error(override_value: Option<&str>) -> ScreenshotError {
    let override_note = match override_value.map(str::trim).filter(|s| !s.is_empty()) {
        Some(value) => format!(" ({SCREENSHOT_TOOL_ENV}={value})"),
        None => String::new(),
    };
    ScreenshotError::NoTool(format!(
        "no screen capture tool is available{override_note}; install grim for Wayland, \
         ImageMagick's import or scrot for X11, or use the built-in screencapture on macOS \
         and PowerShell on Windows. Set {SCREENSHOT_TOOL_ENV} to a capture program on PATH \
         if yours has another name. Capturing also needs a graphical display."
    ))
}

/// Capture the screen and return the path of a real PNG file. The caller owns
/// the file and is responsible for deleting it when it is done with it.
pub fn capture_screenshot() -> Result<PathBuf, ScreenshotError> {
    let override_value = std::env::var(SCREENSHOT_TOOL_ENV).ok();
    capture_screenshot_with(override_value.as_deref())
}

/// The capture path with the environment override passed in explicitly, so the
/// selection can be exercised without changing a process-wide environment.
fn capture_screenshot_with(override_value: Option<&str>) -> Result<PathBuf, ScreenshotError> {
    let tools = tools_for(override_value);
    let mut found_any = false;
    let mut last_failure: Option<ScreenshotError> = None;

    for tool in tools {
        let Some(executable) = find_in_path(tool.name()) else {
            continue;
        };
        found_any = true;
        let path = temp_png_path();
        let result =
            run_capture(&executable, tool, &path).and_then(|()| verify_png(&path, tool.name()));
        match result {
            Ok(()) => return Ok(path),
            Err(error) => {
                let _ = std::fs::remove_file(&path);
                last_failure = Some(error);
            }
        }
    }

    if !found_any {
        return Err(no_tool_error(override_value));
    }
    Err(last_failure.unwrap_or_else(|| no_tool_error(override_value)))
}

fn run_capture(executable: &Path, tool: Tool, path: &Path) -> Result<(), ScreenshotError> {
    let output = Command::new(executable)
        .args(tool.args(path))
        .output()
        .map_err(|e| ScreenshotError::CaptureFailed {
            tool: tool.name().to_string(),
            detail: e.to_string(),
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = [stderr, stdout]
        .into_iter()
        .find(|s| !s.is_empty())
        .unwrap_or_else(|| format!("exit status {}", output.status));
    Err(ScreenshotError::CaptureFailed {
        tool: tool.name().to_string(),
        detail,
    })
}

/// Success from the capture program is not enough: check that a nonempty file
/// with a PNG signature really landed at the path we asked for.
fn verify_png(path: &Path, tool_name: &str) -> Result<(), ScreenshotError> {
    let bytes = std::fs::read(path).map_err(|_| ScreenshotError::NoPng {
        tool: tool_name.to_string(),
        path: path.display().to_string(),
    })?;
    if bytes.len() < PNG_MAGIC.len() || bytes[..PNG_MAGIC.len()] != PNG_MAGIC {
        return Err(ScreenshotError::NoPng {
            tool: tool_name.to_string(),
            path: path.display().to_string(),
        });
    }
    Ok(())
}

fn temp_png_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "xencode-screenshot-{}-{nanos}-{seq}.png",
        std::process::id()
    ))
}

/// Walk `PATH` ourselves rather than shelling out to `which`, matching the
/// approach the voice recorder uses. A directory is accepted only when it
/// holds an executable file of that name.
fn find_in_path(name: &str) -> Option<PathBuf> {
    let search = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&search) {
        #[cfg(windows)]
        let extensions: &[&str] = &[".EXE", ".CMD", ".BAT", ""];
        #[cfg(not(windows))]
        let extensions: &[&str] = &[""];

        for extension in extensions {
            let candidate = directory.join(format!("{name}{extension}"));
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    path.is_file()
        && std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
}

#[cfg(windows)]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::{detect_format, dimensions, ImageFormat};

    #[test]
    fn the_override_names_one_tool_or_refuses_capture() {
        assert_eq!(
            tools_for(Some(" grim ")),
            vec![Tool::Grim],
            "the override wins on any platform"
        );
        assert_eq!(tools_for(Some("scrot")), vec![Tool::Scrot]);
        assert!(tools_for(Some("none")).is_empty());
        assert!(tools_for(Some("disabled")).is_empty());
        assert!(tools_for(Some("a-program-that-does-not-exist")).is_empty());
    }

    #[test]
    fn the_default_selection_uses_grim_first_on_wayland() {
        let tools = default_tools();
        if cfg!(target_os = "linux")
            && std::env::var("WAYLAND_DISPLAY")
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
        {
            assert_eq!(tools.first(), Some(&Tool::Grim));
        }
        assert!(!tools.is_empty(), "every supported platform has a tool");
    }

    /// The no-tool path must not start a process or leave a file behind. An
    /// override naming a program that is not installed forces exactly that.
    #[test]
    fn an_unavailable_capture_tool_is_reported_actionably() {
        let override_value = "xencode-no-such-capture-program";
        assert!(find_in_path(override_value).is_none());
        let error = capture_screenshot_with(Some(override_value)).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(override_value), "{message}");
        assert!(message.contains(SCREENSHOT_TOOL_ENV), "{message}");
        assert!(message.contains("grim"), "{message}");
    }

    #[test]
    fn powershell_quotes_a_path_that_holds_a_quote() {
        let script = powershell_script("C:\\Users\\O'Brien\\shot.png");
        assert!(
            script.contains("'C:\\Users\\O''Brien\\shot.png'"),
            "{script}"
        );
    }

    /// A real capture on this machine: `grim` is installed and a Wayland
    /// display is present. On a machine with no display or capture program the
    /// test says why it cannot run instead of pretending to have captured.
    #[test]
    fn a_real_capture_produces_a_nonempty_png() {
        if find_in_path("grim").is_none()
            || !std::env::var("WAYLAND_DISPLAY")
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
        {
            eprintln!("skipped: no grim and no Wayland display on this host");
            return;
        }
        let path = capture_screenshot_with(None).expect("grim captures a real screenshot");
        let bytes = std::fs::read(&path).expect("the captured file is readable");
        assert!(bytes.len() > PNG_MAGIC.len(), "empty capture");
        assert_eq!(&bytes[..PNG_MAGIC.len()], &PNG_MAGIC);
        assert_eq!(detect_format(&bytes), Some(ImageFormat::Png));
        let (width, height) = dimensions(&bytes, ImageFormat::Png).expect("PNG has IHDR");
        assert!(width > 0 && height > 0, "{width}x{height}");
        eprintln!(
            "captured {} bytes, {width}x{height}, at {}",
            bytes.len(),
            path.display()
        );
        std::fs::remove_file(&path).unwrap();
    }
}
