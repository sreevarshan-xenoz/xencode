//! Google Colab orchestration for xencode.
//!
//! Xencode can serve its OpenAI-compatible inference endpoint from a Colab VM
//! (T4 GPU and friends) and keep a local `ssh -L` port forward to it. This
//! crate owns that bridge lifecycle: the preflight gate (is the `colab` CLI
//! installed, signed in, and is an SSH key ready?), then `up` / `status` /
//! `down` in the same module family.
//!
//! The only supported transport is the official `google-colab-cli`
//! (`colab ssh --proxy-mode`), an authenticated WebSocket SSH route that the
//! Colab CLI itself implements — never a public tunnel. Colab's free tier
//! prohibits ngrok/cloudflared style public tunnels (account suspension), so
//! nothing here ever starts one.
#![forbid(unsafe_code)]

pub mod preflight;

pub use preflight::{preflight, which, Check, PreflightReport};
