//! Real programs in pseudo-terminals, for the orchestrator's tiles (OM-1).
//!
//! A [`Session`] runs one program (a coding agent's own interface, or a
//! shell) in a pseudo-terminal: ConPTY on Windows, a Unix PTY elsewhere. Its
//! output feeds a [`Screen`], which keeps what the program drew and a
//! scrollback, and an [`Attention`] detector, which reads the program's own
//! signals for "needs you". [`draw`] puts a screen into a ratatui buffer.
//!
//! Nothing here interprets what an agent says: the bytes pass through, and
//! the only signals read are the standard ones a terminal program sends on
//! purpose (the bell, and the OSC 9 and OSC 777 notifications).

mod attention;
mod render;
mod screen;
mod session;

pub use attention::{Attention, TileState, IDLE_AFTER};
pub use render::draw;
pub use screen::{Screen, SCROLLBACK};
pub use session::{CommandSpec, Session, Size, TermError};
