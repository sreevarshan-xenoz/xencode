# Xencode Agent Guidelines & Project Directives

## ⚠️ Critical Architecture Rule: Rust-Only

- **The product is Rust**: everything lives under the `rust/` workspace (`rust/crates/*`). The legacy Python stack was deleted — do **NOT** reintroduce Python code, packaging, or Python-based tooling (pip/pytest/ruff/bandit/PyInstaller).
- **Rust First**: All new development, bug fixes, features, TUI improvements, model provider integrations, settings, and CLI tools MUST be written in Rust under the `rust/` directory (`rust/crates/*`).
- When inspecting or fixing functionality (such as model selection, settings panel, Ollama integration, etc.), always target the Rust implementation (`rust/crates/xencode-tui-rs`, `rust/crates/xencode-models-rs`, `rust/crates/xencode-providers-rs`, `rust/crates/xencode-config-rs`, etc.).

## 🔄 Commit Rule: Atomic Git Commits

- **Commit Each Change Before Doing Next Changes**: Every logical task, feature, or bug fix MUST be committed to git immediately upon completion and verification before proceeding to the next change. Never accumulate multiple unrelated changes in the working tree without committing each step first.

## 📚 Docs Rule: Periodic Documentation Updates

- **Docs ride with features**: every user-facing change (new command, flag, TUI command, behavior change, install/infra change) MUST update the affected manuals in the same pass — `README.md`, `QUICK_START.md`, `CLI_GUIDE.md` as applicable — plus `NEXT_PLAN_TASKS.md` (check off finished items, fix stale counts) and `CHANGELOG.md` (Unreleased entries).
- **No fiction in manuals**: document only subcommands, flags, and behavior verified against the implementation (`--help` output, enum definitions, live runs). Never invent commands.
- **Counts stay current**: crate/test counts in `README.md` and `NEXT_PLAN_TASKS.md` must match `cargo test --workspace` at the time of the docs commit.
- **Sweep every few features**: after every 2–3 feature commits (or weekly, whichever comes first), re-check the entry-point docs for drift and correct everything in one docs commit.

