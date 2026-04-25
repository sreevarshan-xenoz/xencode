# Rust Migration Status

Branch: `total-migiration-rust`

## Current Slice

The migration has started with a dependency-light Rust workspace and baseline
benchmark tooling.

## Added

- `rust/Cargo.toml`: Cargo workspace for Rust migration work.
- `rust/crates/xencode-core-rs`: first core Rust crate.
- `rust/crates/xencode-cli`: prototype Rust CLI binary.
- `scripts/baseline_benchmarks.py`: Python baseline benchmark runner.
- `tests/rust/test_workspace_scan_parity.py`: parity test comparing Rust scan output with the Python baseline.

## First Rust Capability

`xencode-core-rs` can scan a workspace and return stable, relative entries with
file type and byte-size metadata. It skips noisy directories such as `.git`,
`target`, `node_modules`, virtualenvs, and Python caches by default.

Prototype command:

```powershell
cd rust
cargo run -p xencode-cli -- scan .. --max-depth 2
```

## Local Verification Notes

Rust is not installed in the current environment, so `cargo test` and
`cargo run` could not be executed here yet. Once Rust is installed, run:

```powershell
cd rust
cargo test
cargo run -p xencode-cli -- scan .. --max-depth 2
```

Python launcher availability also appears limited in this shell, so benchmark
execution may need a normal Python install on PATH:

```powershell
python scripts/baseline_benchmarks.py --root . --iterations 5
```

One local benchmark run completed with `--iterations 1`:

- `python_workspace_scan`: 64.82 ms, 850 files, 212 directories.
- `python_import_xencode_core`: timed out after 20 seconds.
- `python_cli_help`: exited with code 1 after 2933.19 ms.
