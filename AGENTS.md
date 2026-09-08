# Xencode Agent Guidelines & Project Directives

## ⚠️ Critical Architecture Rule: Migration to Rust

- **Complete Rust Migration**: The project is **completely migrating to Rust**. 
- **NO Python Development**: Do **NOT** develop, add features, or write new code in Python (`xencode/` or root Python files).
- **Rust First**: All new development, bug fixes, features, TUI improvements, model provider integrations, settings, and CLI tools MUST be written in Rust under the `rust/` directory (`rust/crates/*`).
- When inspecting or fixing functionality (such as model selection, settings panel, Ollama integration, etc.), always target the Rust implementation (`rust/crates/xencode-tui-rs`, `rust/crates/xencode-models-rs`, `rust/crates/xencode-providers-rs`, `rust/crates/xencode-config-rs`, etc.).

## 🔄 Commit Rule: Atomic Git Commits

- **Commit Each Change Before Doing Next Changes**: Every logical task, feature, or bug fix MUST be committed to git immediately upon completion and verification before proceeding to the next change. Never accumulate multiple unrelated changes in the working tree without committing each step first.

