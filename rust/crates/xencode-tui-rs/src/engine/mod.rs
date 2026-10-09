//! The engine (EN-1): every agent action a window asks for, and everything
//! the agent loop reports back, as typed messages. In this stage the engine
//! runs inside the terminal app; `EN-2` carries the same messages over a
//! local socket to an engine in its own process.

pub mod proto;
