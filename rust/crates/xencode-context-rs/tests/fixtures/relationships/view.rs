//! A second trait implementation, in another module.

use crate::traits::Render;

pub struct TextView;

impl Render for TextView {
    fn render(&self) -> String {
        String::new()
    }
}
