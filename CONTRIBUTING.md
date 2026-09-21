# Contributing to Xencode

Thank you for your interest in contributing to Xencode! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Reporting](#issue-reporting)
- [Security](#security)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and inclusive community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xencode.git
   cd xencode
   ```

3. **Build the workspace** (from the `rust/` directory — everything lives in one
   Cargo workspace there):
   ```bash
   cd rust
   cargo build
   ```

## Development Setup

### Required Dependencies

- Rust (stable toolchain, edition 2021) — `rustup` recommended
- Git
- Ollama and/or llama.cpp (only needed to exercise local model inference)

### Optional Dependencies

- Docker (for the collaboration server image)

### Running the App Under Development

```bash
cd rust
cargo run -p xencode-cli -- --help      # CLI
cargo run -p xencode-cli -- tui         # TUI
```

The CLI reads and writes `~/.xencode/config.json`. Set `XCODE_CONFIG_DIR` to a
scratch directory to keep experiments (config, audit log, conversation memory)
out of your real home directory — and to keep test runs hermetic.

### Running Tests

```bash
cd rust
cargo test --workspace            # all crates
cargo test -p xencode-tui-rs      # one crate
```

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Rust toolchain version (`rustc --version`)
   - OS version
   - Steps to reproduce
   - Expected vs actual behavior
   - Code snippets if applicable

### Suggesting Features

1. Open a GitHub issue with the "enhancement" label
2. Describe the use case
3. Explain why this feature would be valuable
4. Provide examples if possible

### Submitting Code

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write tests** for new functionality

4. **Run the full gate** from `rust/` — all three must be clean:
   ```bash
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings -A clippy::format-in-format-args
   cargo fmt --all --check
   ```

5. **Update the manuals** in the same pass. A user-facing change (new command,
   flag, TUI slash command, behavior change) is not finished until `README.md`,
   `QUICK_START.md`, `CLI_GUIDE.md` and `CHANGELOG.md` reflect it, and
   `NEXT_PLAN_TASKS.md` records the step.

6. **Commit your changes** atomically — one logical change per commit, with a
   clear message (see below)

7. **Push and create a PR**

## Coding Standards

### Rust Style

- `cargo fmt` is authoritative for formatting; CI fails on `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings` must be clean
  (`clippy::format-in-format-args` is allowed workspace-wide)
- Prefer `thiserror`-style typed errors over stringly-typed `Result`s; the
  workspace crates each own their error enum
- `unwrap()`/`expect()` are fine in tests, not in product paths that can hit a
  missing file, a network failure or a terminal that isn't there

### Code Organization

- The product is Rust only, under the `rust/` workspace. Do not reintroduce a
  Python stack, `requirements.txt`, or Python linting/test tooling.
- Put library code in the crate that owns the concern
  (`xencode-config-rs`, `xencode-models-rs`, `xencode-providers-rs`, `xencode-mcp-rs`, …)
  and keep `xencode-cli`/`xencode-tui-rs` as thin composition layers
- Keep functions small and focused; use `mod.rs`-free module files
- Write doc comments (`///`) on public items — the module-level `//!` header
  should say *why* the module exists, not restate what each function does

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/), scoped to
the crate or area the change touches:

```
feat(agent): provider fallback chain
fix(tui): hermetic tests — slash commands leave memory
docs: honesty sweep across manuals
test(mcp): cover server start/stop routing
refactor(config): move URL validation into a helper
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Security Best Practices

- **Never commit secrets.** API keys live in the `api_keys` object inside
  `~/.xencode/config.json`, which is gitignored. `xencode config set` refuses
  key names, so edit the file directly and restrict it (`chmod 600`) — there is
  no encryption layer, so file permissions are the control.
- Keep the agent's approval gate intact: tool classes that edit files or run
  commands require approval unless the user has explicitly lowered it.
- Validate all user inputs at boundaries (CLI args, config values, MCP results)
- The collaboration server must authenticate every mutation with a bearer token
  and record joins, mutations and denials in its audit log

## Testing

### Test Layers

- **Unit tests**: `#[cfg(test)] mod tests` inside each crate, next to the code
  they exercise
- **Integration tests**: `crates/<crate>/tests/*.rs`
- **Async tests**: `#[tokio::test]` for anything touching providers or the agent loop

Tests must not depend on the developer's home directory, a running Ollama, or
the network. Use `App::for_tests()` for TUI tests (non-persistent conversation
memory), an explicit `XCODE_CONFIG_DIR` pointing at a temp dir when the config
loader is involved, and `ConversationMemory::new(n)` over
`with_persistence` in tests.

```rust
#[tokio::test]
async fn slash_commands_stay_out_of_conversation_memory() {
    let mut app = App::for_tests();
    let (tx, _rx) = mpsc::unbounded_channel::<String>();

    let before = app.memory.get_context(100_000).len();
    app.set_chat_text("/mcp status");
    app.submit_message(tx);
    assert_eq!(app.memory.get_context(100_000).len(), before);
}
```

### Running Tests

```bash
cd rust
cargo test --workspace                        # everything
cargo test -p xencode-providers-rs            # one crate
cargo test -p xencode-tui-rs focus            # tests matching a name
```

## Pull Request Guidelines

### PR Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings -A clippy::format-in-format-args` is clean
- [ ] `cargo fmt --all --check` is clean
- [ ] Tests added/updated for new behavior
- [ ] Manuals updated (`README.md`, `QUICK_START.md`, `CLI_GUIDE.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md`)
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with `main`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No security issues introduced
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly
4. **Squash commits** if requested

## Issue Reporting

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information needed

### Issue Templates

Use the provided GitHub issue templates for:
- Bug reports
- Feature requests
- Security issues

## Security

### Reporting Security Issues

**Do not open public issues for security vulnerabilities.**

Email security concerns to: security@xenoz.com

### Security Best Practices for Contributors

1. Never commit credentials or secrets
2. Keep API keys in `~/.xencode/config.json` (`chmod 600`) or environment
   variables — never in the repo
3. Validate all inputs
4. Follow secure coding guidelines
5. Keep tool approvals on by default for anything that edits files or runs
   commands

## Questions?

- Check existing [documentation](docs/)
- Search [closed issues](https://github.com/sreevarshan-xenoz/xencode/issues?q=is%3Aissue+is%3Aclosed)
- Join our [Discord](https://discord.com/invite/d9ewZkWPTP)

Thank you for contributing to Xencode! 🎉
