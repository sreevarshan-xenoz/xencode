//! The resting badge: a round logo whose colour and motion carry the most
//! urgent state across all running sessions.

use std::path::PathBuf;
use std::time::Duration;

use gpui::{
    div, point, prelude::*, px, rgb, rgba, size, App, Bounds, Context, MouseButton, Task, Window,
    WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowHandle, WindowKind,
    WindowOptions,
};
use xencode_live_rs::badge::{BadgeView, Shown};

use crate::card::{self, Card};
use crate::position::{self, Saved};

/// One animation step; files are re-read every `REFRESH_EVERY` steps.
const STEP: Duration = Duration::from_millis(150);
const REFRESH_EVERY: usize = 7;

pub struct Badge {
    live: Option<PathBuf>,
    view: BadgeView,
    /// Sessions whose failure the person has hovered over.
    seen: Vec<String>,
    step: usize,
    card: Option<WindowHandle<Card>>,
    position_file: Option<PathBuf>,
    last_origin: Option<Saved>,
    _ticker: Task<()>,
}

impl Badge {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let ticker = cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor().timer(STEP).await;
            if this
                .update_in(cx, |badge, window, cx| badge.on_step(window, cx))
                .is_err()
            {
                break;
            }
        });
        let mut badge = Badge {
            live: xencode_live_rs::live_dir().ok(),
            view: xencode_live_rs::badge::view(&[], 0),
            seen: Vec::new(),
            step: 0,
            card: None,
            position_file: position::settings_file(),
            last_origin: None,
            _ticker: ticker,
        };
        badge.refresh();
        badge
    }

    fn refresh(&mut self) {
        let reads = match &self.live {
            Some(dir) => xencode_live_rs::read_all(dir),
            None => Vec::new(),
        };
        self.view = xencode_live_rs::badge::view(&reads, xencode_live_rs::now_secs());
        // A session that is gone, or no longer failed, no longer needs remembering.
        self.seen.retain(|id| {
            self.view
                .rows
                .iter()
                .any(|r| &r.session_id == id && r.shown == Shown::Failed)
        });
    }

    fn shown(&self) -> Shown {
        self.view.shown_after(&self.seen)
    }

    fn animating(&self) -> bool {
        matches!(self.shown(), Shown::Working | Shown::NeedsYou)
    }

    fn on_step(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.step = self.step.wrapping_add(1);
        // The hover listener can miss the mouse leaving when it jumps away in
        // one move (to another monitor, say); the window's own hover state
        // still knows, so the card closes on the next step. On macOS that
        // state means "active", which this unfocused badge never is.
        if self.card.is_some()
            && cfg!(any(target_os = "windows", target_os = "linux"))
            && !window.is_window_hovered()
        {
            self.close_card(cx);
            cx.notify();
        }
        let mut changed = false;
        if self.step.is_multiple_of(REFRESH_EVERY) {
            let before = self.view.clone();
            self.refresh();
            changed = before != self.view;
            self.remember_position(window);
            if changed {
                self.update_card(cx);
            }
        }
        if changed || self.animating() {
            cx.notify();
        }
    }

    /// Save where the person dragged the badge to, once it has moved.
    fn remember_position(&mut self, window: &Window) {
        let origin = window.bounds().origin;
        let now = Saved {
            x: f32::from(origin.x),
            y: f32::from(origin.y),
        };
        if self.last_origin.is_some_and(|before| before != now) {
            if let Some(file) = &self.position_file {
                let _ = position::save(file, now);
            }
        }
        self.last_origin = Some(now);
    }

    fn on_hover(&mut self, hovered: bool, window: &mut Window, cx: &mut Context<Self>) {
        if hovered {
            for row in &self.view.rows {
                if row.shown == Shown::Failed && !self.seen.contains(&row.session_id) {
                    self.seen.push(row.session_id.clone());
                }
            }
            self.open_card(window, cx);
        } else {
            self.close_card(cx);
        }
        cx.notify();
    }

    fn open_card(&mut self, window: &Window, cx: &mut Context<Self>) {
        if self.card.is_some() {
            return;
        }
        let badge = window.bounds();
        let screen = cx.primary_display().map(|d| d.bounds()).unwrap_or(badge);
        let height = card::height(self.view.rows.len(), self.view.skipped.len());
        let badge_x = f32::from(badge.origin.x);
        let screen_mid = f32::from(screen.origin.x) + f32::from(screen.size.width) / 2.0;
        // Open towards the middle of the screen, never over the badge.
        let x = if badge_x > screen_mid {
            badge_x - card::WIDTH - position::MARGIN
        } else {
            badge_x + position::SIZE + position::MARGIN
        };
        let top = f32::from(screen.origin.y);
        let bottom = top + f32::from(screen.size.height);
        let y = (f32::from(badge.origin.y) + position::SIZE / 2.0 - height / 2.0)
            .clamp(top, (bottom - height).max(top));
        let rows = self.view.rows.clone();
        let skipped = self.view.skipped.clone();
        let opened = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds {
                    origin: point(px(x), px(y)),
                    size: size(px(card::WIDTH), px(height)),
                })),
                titlebar: None,
                kind: WindowKind::PopUp,
                focus: false,
                is_movable: false,
                is_resizable: false,
                is_minimizable: false,
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            },
            |window, cx| {
                crate::border::remove_frame(window);
                cx.new(|_| Card { rows, skipped })
            },
        );
        self.card = opened.ok();
    }

    fn close_card(&mut self, cx: &mut App) {
        if let Some(card) = self.card.take() {
            let _ = card.update(cx, |_, window, _| window.remove_window());
        }
    }

    fn update_card(&mut self, cx: &mut App) {
        if let Some(card) = self.card {
            let rows = self.view.rows.clone();
            let skipped = self.view.skipped.clone();
            let _ = card.update(cx, |card, _, cx| {
                card.rows = rows;
                card.skipped = skipped;
                cx.notify();
            });
        }
    }
}

impl Render for Badge {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let shown = self.shown();
        let pulse = 0.8 + 0.2 * ((self.step as f32) * 0.5).sin();
        let (fill, opacity, label) = match shown {
            Shown::Nothing => (0x4b5563, 0.6, "x"),
            Shown::Working => (0x2563eb, 1.0, "x"),
            Shown::NeedsYou => (0xf59e0b, pulse, "?"),
            Shown::Finished => (0x16a34a, 1.0, "✓"),
            Shown::Failed => (0xdc2626, 1.0, "!"),
        };
        let mut badge = div()
            .id("badge")
            .relative()
            .size(px(position::SIZE))
            .rounded_full()
            .bg(rgb(fill))
            .opacity(opacity)
            .flex()
            .items_center()
            .justify_center()
            .text_color(rgb(0xffffff))
            .text_lg()
            .child(label)
            .on_hover(
                cx.listener(|this, hovered: &bool, window, cx| this.on_hover(*hovered, window, cx)),
            )
            // Drag it anywhere; right-click closes it. On Windows the OS moves a
            // window by its drag area (GPUI's start_window_move does nothing
            // there); elsewhere start_window_move does it.
            .window_control_area(WindowControlArea::Drag)
            .on_mouse_down(MouseButton::Left, |_, window, _| window.start_window_move())
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, _, window, cx| {
                    // Save a position the person just dragged to: the regular
                    // save waits for the next one-second refresh.
                    this.remember_position(window);
                    cx.quit();
                }),
            );
        if shown == Shown::Working {
            // Eight dots around the edge, one lit, stepping round: a turning ring.
            let lit = self.step % 8;
            for i in 0..8 {
                let angle = (i as f32) * std::f32::consts::FRAC_PI_4 - std::f32::consts::FRAC_PI_2;
                let centre = position::SIZE / 2.0;
                let radius = centre - 4.0;
                badge = badge.child(
                    div()
                        .absolute()
                        .left(px(centre + radius * angle.cos() - 2.0))
                        .top(px(centre + radius * angle.sin() - 2.0))
                        .size(px(4.0))
                        .rounded_full()
                        .bg(if i == lit {
                            rgba(0xffffffff)
                        } else {
                            rgba(0xffffff55)
                        }),
                );
            }
        }
        if self.view.stale_dot {
            badge = badge.child(
                div()
                    .absolute()
                    .top(px(0.0))
                    .right(px(0.0))
                    .size(px(10.0))
                    .rounded_full()
                    .bg(rgb(0x9ca3af))
                    .border_1()
                    .border_color(rgb(0x111827)),
            );
        }
        badge
    }
}
