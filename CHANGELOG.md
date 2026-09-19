# Changelog

All notable changes to the Xencode project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Chat input history & autocomplete: Alt+Up/Down recalls sent prompts (draft stashed and restored, adjacent duplicates skipped), Tab completes slash commands — `/init`, `/ctx`, `/advise`, `/bytebot` — with the command list surfaced in the help overlay and as a toast on a lone `/` (Milestone E5)
- Multiline chat input: the input box is now a real text area — Enter sends, Alt+Enter (or Ctrl+J) inserts a newline, arrows move inside multi-line drafts (Milestone E5)
- `light` TUI theme (8th palette), selectable via Settings ←/→ or `active_theme = "light"` in config; theme cycling now runs through one shared `THEME_NAMES` list instead of three duplicated arrays, and the settings Theme row shows dots for all themes (Milestone E4)

### Changed
- Every panel color renders through new semantic `ThemeColors` slots (`success`/`warning`/`danger`/`info`/`accent_secondary`): 63 hardcoded ANSI constants in the renderer are gone, so themes can actually restyle severities, diffs and status badges (Milestone E4)
- Body panel layout has one source of truth: `ui::body_chunks` feeds both the renderer and mouse focus hit-testing (`ui::body_hit_test`), replacing the duplicated 20%/70% magic numbers in the event loop (Milestone E4)
- Settings panel navigation is bound to a real row list: new `focus::SETTINGS_ROWS` feeds the panel labels, section ranges, and the ↑↓/wheel bounds, replacing the hardcoded `14` (Milestone E4)
- Removed legacy Node wrapper (`package.json`, `package-lock.json`, `bin/xencode.js`) and completed-migration docs (`docs/RUST_MIGRATION_PLAN.md`, `docs/RUST_MIGRATION_STATUS.md`); the product is Rust-only under `rust/`.

### Added
- TUI toast notification layer: file-watch warnings now surface as transient top-right overlays (6s TTL, deduped) instead of polluting chat history as fake system lines (Milestone E3)
- TUI keybinding help overlay (`?`/`F1`): modal panel-aware keymap generated from the real key handler, with scroll and status-bar discoverability (Milestone E3)
- TUI markdown rendering for assistant chat: fenced code blocks with language labels (correct even mid-stream), headings, bullets/ordered lists, block quotes, rules, inline `code`/**bold**/*italic* (Milestone E3)
- Image input pipeline: magic-byte format detect, header dimensions, data URLs (`analysis-rs` images module), `analyze` CLI inventory, `MessageContent` parts with per-backend rendering (Ollama/Anthropic/Gemini/OpenAI-compatible), TUI attach sends images as message parts
- Web extraction for research: timeout/capped fetch with content-type gate, pure HTML→text (scripts/styles stripped, entities decoded), `FetchedPage` record, `fetch` CLI subcommand
- PR-level diff triage: rename-aware numstat parsing and capped per-file diffs (`gitinfo`), `review` CLI command with working-tree analysis and documented JSON
- Document parsing into context: PDF (pdf-extract) and DOCX (zip + w:t runs) text extraction with size/char caps, TUI attach inlines extracted text with explicit skip notes
- Workspace RBAC + audit log: Admin-gated membership, self-leave, last-admin guard, sequenced audit events including denials
- TUI PR review dashboard: per-file diff browsing (`Ctrl+Y`, Feature Navigator entry) over rename-aware numstat + capped diffs, base toggle HEAD<->main, scrollable diff pane

### Fixed
- TUI: every scroll offset (chat, code review, provider health, security findings, help overlay) is re-clamped when the terminal resizes, so shrinking then growing back no longer re-exposes stale scroll-past-the-end blank space; render and clamp now share one set of line builders (E6-02)
- TUI: text fields (git commit message, ByteBot command, settings URL edit) are now fully typable — space and the letters `i / m s q ?` no longer trigger shortcuts while typing; Left arrow moves the cursor instead of deleting a character; VoiceInterface `m` mute reachable again (E2-06)
- TUI (Milestone E2): ByteBot Enter executes the typed command instead of clobbering it with history; `git commit` runs off the UI thread with the result reported as a chat line; dead `FocusArea::Terminal` removed; unused `file_scroll_offset` deleted; provider-health/security/code-review scroll clamped (no more blank render past the end) and CodeReview output made scrollable; mouse wheel wired for Settings/ModelSelector/CustomModels/CodeReview
- Docs: corrected stale crate/test counts (`NEXT_PLAN.md`, `docs/ROADMAP.md` said 331 tests; real count verified at 421/13 crates via `cargo test --workspace`), removed Python-era fiction from `docs/USER_MANUAL.md` and `docs/INSTALL_MANUAL.md` (dead `xencode.sh`/pip instructions, missing HTML reference)
- Context budget: state/git tiers emitted only when margin-admitted; tail truncation hard-cuts single-line overflow; preview compaction flag only when content dropped
- Symbol graph: brace-group import expansion and crate-root fallback for 2018 paths; broken-import checks per anchored pair
- TUI: no silent attachment drops, exact watch-dedup, kind|path watch tokens, honest /advise counts, visible save errors
- CLI: full-tree analyze with skip counts and documented dir JSON schema, scan --format json, fetch text cap, strict --json-schema
- Providers: request timeouts (total for single-shot, time-to-first-token for streams), Gemini key via header, NXDOMAIN-only DNS retry direction, tool-call demux for index-less fragments
- Cache: deterministic LRU sequence counter (fixes intermittent eviction flakes), float-tolerant legacy seeding test
- Real-time workspace watcher with debounced proactive warnings for tracked/attached/open files
- Repository insights: dependency cycles, hub files, orphans, broken imports, affected dependents (`advise` module + `/advise` TUI command with dependents in watch warnings)
- Rust CI gates (fmt, clippy `-D warnings`, full workspace suite), Rust binary GitHub Releases, Rust Docker/runtime and installers
- Enhanced Security Scanning with Bandit Integration
- Comprehensive vulnerability database with CVE patterns
- OWASP Top 10 vulnerability detection
- Multi-language security analysis (Python, JavaScript, Java)
- Security report generation (summary, detailed, executive)
- Automated risk assessment and compliance scoring
- 50+ Bandit security rules with CWE mappings
- Async security scanning for performance
- Demo application for security scanning capabilities
- Comprehensive test suite (21 tests) for security features

### Removed
- Legacy Python stack: `xencode/` package, pytest suite, examples, benchmarks, deployment helpers, packaging configs, and the local venv (Rust workspace under `rust/` is the product)
- Python install paths (pip/venv/PyInstaller) from `install.sh`/`install.ps1`, the Python Dockerfile stages, and pytest/ruff/bandit CI jobs

### Enhanced
- Security analyzer with multiple scanning methods
- Pattern-based vulnerability detection
- Dependency vulnerability checking
- Security metrics calculation and reporting
- Integration with existing code analysis workflow

### Security
- Code injection detection (eval, exec, dynamic execution)
- SQL injection pattern matching
- Cross-site scripting (XSS) vulnerability detection
- Path traversal security checks
- Weak cryptography identification
- Hardcoded secrets detection
- Unsafe deserialization checks
- Command injection vulnerability scanning

## [2.1.0] - 2026-03-30

### Added
- Dynamic feature API route mounting with auth protection for runtime feature endpoints
- Missing feature hook implementations for CLI, TUI, and API integration across core feature modules
- API auth regression tests for protected routes and dynamically mounted feature routes
- Integration smoke tests for health and info endpoints
- Release and deployment validation scripts for version consistency and deployment secrets

### Changed
- Hardened collaborative CLI async execution and TUI panel update safety in unmounted contexts
- Enforced JWT auth on code analysis and document routers
- Improved CI/CD workflows with marker-matrix test jobs, collect-only precheck, and deploy secret preflight
- Updated Kubernetes secret handling to use templated secrets manifest

## [3.0.0] - 2024-10-15

### Added
- Phase 2 Core Infrastructure
- Intelligent Model Selection Engine
- Advanced Caching System with hybrid storage
- Smart Configuration Management
- Advanced Error Handling Framework
- System Health Monitoring and Coordination
- Comprehensive test suite (24 tests)
- Interactive demo system
- Production-ready deployment scripts

### Performance
- 99.9% performance improvement with advanced caching
- Sub-millisecond response times for cached operations
- Automated hardware optimization
- Real-time system health monitoring

### Reliability
- 95%+ automatic error recovery rate
- Enterprise-grade resilience framework
- Intelligent fallback strategies
- Context-aware diagnostic messaging

### Developer Experience
- Zero-configuration setup
- Interactive deployment wizard
- Hot-reload configuration updates
- Comprehensive documentation

## [2.0.0] - 2024-09-15

### Added
- Core AI assistant functionality
- Basic model management
- Simple caching system
- Configuration management
- CLI interface

### Changed
- Improved response handling
- Enhanced model selection
- Better error messages

## [1.0.0] - 2024-08-15

### Added
- Initial release
- Basic AI chat functionality
- Ollama integration
- Simple CLI interface