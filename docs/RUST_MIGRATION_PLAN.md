# Rust Migration Plan

Branch: `total-migiration-rust`

## Goal

Migrate Xencode from a Python-first codebase to a Rust-first architecture to improve startup time, runtime performance, packaging reliability, memory usage, and long-running TUI responsiveness.

Rust will not automatically make every feature faster by itself. The biggest wins should come from moving CPU-heavy, IO-heavy, and frequently called paths first, while keeping model-provider network calls and plugin surfaces stable during the transition.

## Guiding Principles

- Keep the existing Python product usable throughout the migration.
- Replace one subsystem at a time behind stable interfaces.
- Measure before and after each migration phase.
- Avoid rewriting experimental or low-traffic features before core paths are proven.
- Keep compatibility with current config files, workspace layout, and user workflows.
- Prefer Rust libraries with strong maintenance and cross-platform support.

## Target Architecture

- `xencode-core-rs`: Rust crate for core workspace, config, caching, model routing, command execution, and shared domain types.
- `xencode-cli`: Rust CLI binary replacing Python command entry points.
- `xencode-tui`: Rust TUI binary, likely using `ratatui` or another mature terminal UI stack.
- `xencode-api`: Rust service layer where long-running HTTP/WebSocket workloads benefit from stronger concurrency.
- Python compatibility shim during transition for legacy features and plugin integrations.

## Phase 0: Baseline And Boundaries

1. Add benchmarks for current Python startup time, chat request latency, file indexing, cache reads/writes, and TUI responsiveness.
2. Map existing modules into migration buckets:
   - Core data and config
   - CLI commands
   - TUI widgets
   - Model providers
   - RAG/indexing
   - Agentic workflows
   - API/server
   - Plugins
3. Define stable JSON schemas for cross-language boundaries.
4. Create a migration tracking board or checklist in `docs/`.

## Phase 1: Rust Core Library

1. Create a Cargo workspace under `rust/`.
2. Implement shared domain models in Rust:
   - Workspace metadata
   - Model/provider definitions
   - Chat messages
   - Tool execution results
   - Config and settings
3. Add Python bindings with `pyo3` only where Python still needs to call Rust.
4. Replace Python hot-path helpers with Rust equivalents:
   - File scanning
   - Path filtering
   - Cache serialization
   - Git/diff parsing helpers
5. Add parity tests comparing Python and Rust outputs.

## Phase 2: CLI Migration

1. Build a Rust CLI with `clap`.
2. Recreate high-value commands first:
   - `query`
   - `models`
   - `tui` launcher
   - file/context commands
   - config commands
3. Preserve current command names and flags where possible.
4. Keep Python fallback commands for incomplete features.
5. Add integration tests for command output and exit codes.

## Phase 3: Provider And Chat Runtime

1. Move provider abstraction to Rust with async `tokio`.
2. Implement providers in this order:
   - Ollama/local HTTP
   - OpenRouter
   - Qwen
   - Gemini CLI bridge
   - Google Gemini API
3. Add unified retry, timeout, streaming, and error handling.
4. Preserve existing config key names.
5. Add streaming tests using mock servers and fake CLI binaries.

## Phase 4: TUI Migration

1. Rebuild the main TUI shell in Rust.
2. Port panels incrementally:
   - Chat panel
   - File explorer
   - Terminal panel
   - Model selector
   - Settings panel
   - Git/status panel
3. Keep feature panels behind a compatibility bridge until rewritten.
4. Use async channels for chat streaming and background tasks.
5. Add snapshot tests for layout and keyboard handling where practical.

## Phase 5: RAG, Indexing, And Analysis

1. Move indexing and file traversal to Rust.
2. Evaluate Rust-native vector/index options or keep existing vector DBs behind a stable interface.
3. Port code-analysis primitives that are deterministic and performance-sensitive.
4. Keep AI-dependent analysis workflows provider-agnostic.
5. Benchmark large-repo indexing and incremental updates.

## Phase 6: Server And Collaboration

1. Replace Python HTTP/WebSocket server with Rust service.
2. Use `axum` or `actix-web`.
3. Port auth, workspace, collaboration, and monitoring routes.
4. Preserve API contracts with compatibility tests.
5. Add load tests for WebSocket collaboration sessions.

## Phase 7: Plugin And Extension Strategy

1. Define a stable plugin protocol using JSON-RPC, MCP-style tools, or process-based plugins.
2. Support existing Python plugins as external processes during transition.
3. Add Rust plugin SDK after core contracts settle.
4. Document how plugins declare permissions, inputs, outputs, and lifecycle hooks.

## Phase 8: Packaging And Release

1. Produce single-file binaries for Windows, macOS, and Linux.
2. Keep Python package release only as a compatibility wrapper until deprecation.
3. Add installer updates for:
   - Rust binary install
   - config migration
   - PATH setup
4. Add release smoke tests for each platform.
5. Define deprecation timeline for Python entry points.

## Performance Targets

- CLI startup under 100 ms for simple commands.
- TUI first render under 300 ms on a normal workspace.
- Large workspace file scan at least 3x faster than current Python baseline.
- Chat streaming overhead below provider latency noise.
- Memory usage reduced for long-running TUI sessions.

## Risks

- A full rewrite can stall if parity tests are missing.
- TUI behavior may regress without snapshot or interaction tests.
- Python plugin compatibility needs a deliberate bridge.
- Some bottlenecks are model/network-bound and will not improve much from Rust.
- Windows terminal behavior must be tested early, not at the end.

## Immediate Next Steps

1. Add baseline benchmark scripts. Started in `scripts/baseline_benchmarks.py`.
2. Create `rust/` Cargo workspace. Started in `rust/`.
3. Implement Rust config and workspace scanning prototypes. Workspace scanning started in `xencode-core-rs`.
4. Add Python/Rust parity tests for the first migrated helpers.
5. Decide whether the first production Rust binary should be the CLI or a core helper library.
