# Common tasks, run from the repository root with `just <recipe>`
# (https://github.com/casey/just). Each recipe is the same command CI runs.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# List the recipes.
default:
    @just --list

# Build the xencode binary (debug).
build:
    cargo build --manifest-path rust/Cargo.toml -p xencode-cli

# Build the release binary.
release:
    cargo build --manifest-path rust/Cargo.toml --release -p xencode-cli

# Run the whole test suite.
test:
    cargo test --manifest-path rust/Cargo.toml --workspace --no-fail-fast

# Check formatting and run clippy, as CI does.
lint:
    cargo fmt --manifest-path rust/Cargo.toml --all -- --check
    cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings

# Format the code.
fmt:
    cargo fmt --manifest-path rust/Cargo.toml --all

# Measure line coverage; CI fails below 80%. Needs cargo-llvm-cov.
coverage:
    cargo llvm-cov --manifest-path rust/Cargo.toml --workspace --summary-only --fail-under-lines 80

# Regenerate the shell completions and man page that CI checks. Unix only:
# Windows PowerShell 5.1 would write the files as UTF-16.
[unix]
docs: build
    ./rust/target/debug/xencode generate completions --shell fish > docs/completions/xencode.fish
    ./rust/target/debug/xencode generate completions --shell zsh > docs/completions/xencode.zsh
    ./rust/target/debug/xencode generate completions --shell bash > docs/completions/xencode.bash
    ./rust/target/debug/xencode generate man > docs/man/xencode.1

# Remove the build folder, which grows with every rebuild.
clean:
    cargo clean --manifest-path rust/Cargo.toml
