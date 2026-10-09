//! The card that slides out while the badge is hovered: one row per running
//! session, with the state always written in words.

use gpui::{div, prelude::*, px, rgb, Context, FontWeight, Window};
use xencode_live_rs::badge::{Row, Shown};

pub const WIDTH: f32 = 380.0;
const ROW_HEIGHT: f32 = 80.0;
const CHROME: f32 = 60.0;
const SKIPPED_HEIGHT: f32 = 20.0;

pub struct Card {
    pub rows: Vec<Row>,
    pub skipped: Vec<String>,
}

/// The card's height for this many rows and unreadable files.
pub fn height(rows: usize, skipped: usize) -> f32 {
    CHROME + ROW_HEIGHT * rows.max(1) as f32 + SKIPPED_HEIGHT * skipped as f32
}

fn state_colour(shown: Shown, responding: bool) -> u32 {
    if !responding {
        return 0x9ca3af;
    }
    match shown {
        Shown::NeedsYou => 0xf59e0b,
        Shown::Failed => 0xef4444,
        Shown::Working => 0x60a5fa,
        Shown::Finished => 0x22c55e,
        Shown::Nothing => 0x9ca3af,
    }
}

impl Render for Card {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = div()
            .size_full()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .bg(rgb(0x111827))
            .border_1()
            .border_color(rgb(0x374151))
            .rounded_lg()
            .text_color(rgb(0xe5e7eb))
            .text_sm()
            .child(
                div()
                    .font_weight(FontWeight::BOLD)
                    .child("xencode sessions"),
            );
        if self.rows.is_empty() {
            body = body.child(
                div()
                    .text_color(rgb(0x9ca3af))
                    .child("No xencode session is running."),
            );
        }
        for row in &self.rows {
            let colour = state_colour(row.shown, row.responding);
            body = body.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .font_weight(FontWeight::BOLD)
                                    .truncate()
                                    .child(row.project.clone()),
                            )
                            .child(div().text_color(rgb(colour)).child(row.state_words.clone())),
                    )
                    .child(div().truncate().text_color(rgb(0xd1d5db)).child(
                        if row.headline.is_empty() {
                            "—".to_string()
                        } else {
                            row.headline.clone()
                        },
                    ))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x9ca3af))
                            .child(format!("{} · {}", row.model, row.since)),
                    ),
            );
        }
        for line in &self.skipped {
            body = body.child(
                div()
                    .text_xs()
                    .truncate()
                    .text_color(rgb(0x6b7280))
                    .child(line.clone()),
            );
        }
        body.w(px(WIDTH))
    }
}
