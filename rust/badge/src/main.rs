//! The floating xencode badge (DK-2): a small round logo at the screen edge
//! that shows what every running xencode session is doing. Hover for a card
//! with one row per session; drag to move; right-click to close.

mod border;
mod card;
mod position;
mod view;

use gpui::{
    point, prelude::*, px, size, App, Bounds, WindowBackgroundAppearance, WindowBounds, WindowKind,
    WindowOptions,
};

fn main() {
    // One badge per user: the second copy finds the lock taken and leaves
    // quietly, so `xencode badge` and the autostart setting can both run it.
    let Some(_lock) = take_lock() else {
        return;
    };
    gpui_platform::application().run(|cx: &mut App| {
        let Some(display) = cx.primary_display() else {
            eprintln!("xencode-badge: no display to show the badge on");
            cx.quit();
            return;
        };
        let screen = display.bounds();
        let saved = position::settings_file().and_then(|file| position::load(&file));
        let at = position::initial(
            saved,
            position::Screen {
                x: f32::from(screen.origin.x),
                y: f32::from(screen.origin.y),
                width: f32::from(screen.size.width),
                height: f32::from(screen.size.height),
            },
        );
        let opened = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds {
                    origin: point(px(at.x), px(at.y)),
                    size: size(px(position::SIZE), px(position::SIZE)),
                })),
                titlebar: None,
                kind: WindowKind::PopUp,
                is_movable: true,
                is_resizable: false,
                is_minimizable: false,
                focus: false,
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            },
            |window, cx| {
                border::remove_frame(window);
                cx.new(|cx| view::Badge::new(window, cx))
            },
        );
        if let Err(e) = opened {
            eprintln!("xencode-badge: could not open the badge window: {e}");
            cx.quit();
        }
    });
}

/// Hold `<live dir>/badge.lock` for the life of the process. `None` when
/// another badge holds it.
fn take_lock() -> Option<std::fs::File> {
    let dir = xencode_live_rs::live_dir().ok()?;
    std::fs::create_dir_all(&dir).ok()?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("badge.lock"))
        .ok()?;
    file.try_lock().ok()?;
    Some(file)
}
