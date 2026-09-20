# Changelog

All notable changes to the Xencode project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The Collaboration Hub gains a real client (Milestone G, G3-01):
  `xencode-tui-rs::collab_client` logs in over HTTP, opens the WebSocket,
  authenticates with the first-frame `auth` token (creating a session via
  `POST /sessions/create` when none is chosen), and translates every server
  frame through pure `frame_to_tokens` into the app's `[COLLAB]` grammar —
  a `members:<json>` snapshot replaces the old per-member simulation, errors
  arrive as `error:`. 30 s ping keepalive with a 60 s liveness deadline;
  no auto-reconnect by design. The simulated session (fake alice/bob/carol
  sync pulses) and its dead state fields are deleted; hub keys land with
  G3-02
- Persistent audit trail for collaboration servers (Milestone G, G1-04):
  `xencode-server-rs::audit::AuditSink` mirrors every `WorkspaceManager`
  event — creations, joins, role changes, and denials (including
  authenticated join refusals: unknown session, session full) — as one
  JSONL line each, appending across restarts. A failed open or write warns
  once and disables the sink; the session plane never notices. Defaults to
  disabled so tests touch no disk; the `--audit-path` flag arrives with the
  CLI server work (G2-01)
- `repo_advise` agentic tool (F3-02): the chat model can now call repository
  insights directly — broken imports, cycles, hubs, orphans — with an optional
  path `filter`, a 40-finding cap and `error:` strings instead of panics; both
  surfaces share a new `advise_from_snapshot` loader in `xencode-context-rs`
  that the insights panel and the CLI now read through as well
- `xencode advise [FILTER] [--json] [--limit 40]` — repository insights
  (broken imports, cycles, hubs, orphans) read straight from the `.xencode`
  snapshot in the current directory; positional filter matches finding file
  paths, `--limit 0` shows all, and a missing index exits 1 with a
  "run /init first" error instead of an empty report
- Insights panel (`Ctrl+L` or Feature Navigator #17, F2-01): deterministic refactor findings from the live symbol graph — broken imports, cycles, hubs, orphans — as a color-coded selectable list; Enter shows the full message, `o` opens the advised file in the editor, `r` recomputes; `/advise` now renders through the same snapshot path, so chat and panel always agree
- The TUI file watcher now keeps the symbol/dep snapshot live (F1-02): modified/removed `.rs` files trigger `refresh_rust_file` before dependent-file warnings are computed, so toasts and `/advise` reflect current imports without another `/init`
- Live symbol-graph refresh (`xencode-context-rs::refresh`, F1-01): `refresh_rust_file` re-extracts one already-indexed `.rs` file from disk (or drops it when deleted), rebuilds `deps.json` and keeps `files.json`/`manifest.json` (sizes, loc, mtimes) consistent — a refreshed snapshot makes the next `/init` report fresh; the watcher now also excludes `.xencode` so snapshot rewrites never loop back into themselves
- `xencode worktree list | add <path> [<branch>] | remove <path>` (D3-04) over the D3-01 helpers; omitted branch lets git name the new branch after the directory, dirty removals stay refused by git and the main checkout is never removable
- Background tasks can run in another directory (D3-03): `TaskManager::start_with_cwd`, an optional `cwd` on the `background_start` tool (echoed back as `in <dir>` in the result), and the context bundle's git summary now lists extra git worktrees with branch and dirty markers
- Worktree panel (`Ctrl+O` or Feature Navigator, D3-02): lists git worktrees with branch, short HEAD and dirty marker (★ main / ⚡ dirty / ◯ clean), `a` walks a two-stage add prompt (path → branch), `d` removes via y/N confirm — the main worktree is refused before any git call and dirty worktrees stay protected by git's own guard; `r` refreshes
- Git worktree helpers (`xencode-context-rs::worktree`, D3-01): pure `parse_worktree_list` for `git worktree list --porcelain` (spaces in paths, detached/locked/prunable/bare) plus explicit-arg `worktree_list`/`worktree_add`/`worktree_remove` shells sharing `gitinfo`'s error reporting
- `xencode tasks` CLI (D2-02): file-backed background-task registry (`xencode-core-rs::tasks_file`) shared through `.xencode/tasks/` — `list [--json]`, `start <cmd> [--name]`, `poll <id> [--lines]`, `stop <id>`, `rm <id>`; tasks survive the starting process (EXIT-trap wrapper records exit codes, output captured to `<id>.out`), status is derived on read from exit file + killed flag + `/proc` liveness, and `rm` refuses still-running tasks
- Background Tasks panel (`Ctrl+K` or Feature Navigator, D2-01): live registry view with per-status colored rows (running / exited / killed), Enter → scrollable output detail for the selected task, `x` stop and `d` remove dispatched through the event channel so keys never block, wheel + resize clamping included
- Agentic background-task tool loop in chat (D1-02): the model can call `background_start` / `background_poll` / `background_stop` (schemas in `xencode-providers-rs::background_tools`), executed client-side against the shared task registry with up to 8 tool rounds per turn; each call and result echoes into chat as a `⚙` system line
- Background task registry (`xencode-core-rs::tasks`, D1-01): pure `TaskStore`/`TaskRecord` (status Running/Exited(code)/Killed, 500-line capped output tail) plus a tokio-backed `TaskManager` that spawns commands through `sh -c`, drains stdout+stderr without blocking polls, and refuses to remove or drop a still-running task's child process
- Chat input history & autocomplete: Alt+Up/Down recalls sent prompts (draft stashed and restored, adjacent duplicates skipped), Tab completes slash commands — `/init`, `/ctx`, `/advise`, `/bytebot` — with the command list surfaced in the help overlay and as a toast on a lone `/` (Milestone E5)
- Multiline chat input: the input box is now a real text area — Enter sends, Alt+Enter (or Ctrl+J) inserts a newline, arrows move inside multi-line drafts (Milestone E5)
- `light` TUI theme (8th palette), selectable via Settings ←/→ or `active_theme = "light"` in config; theme cycling now runs through one shared `THEME_NAMES` list instead of three duplicated arrays, and the settings Theme row shows dots for all themes (Milestone E4)

### Changed
- Manuals tell the truth about team mode (Milestone G, G4-01): the user
  manual gained a Collaboration Hub key table and lost its fabricated
  claims (`0.0.0.0` default, "credential vault… email verification");
  `docs/api_documentation.md`'s Python-era JWT auth matrix was replaced by
  the real route table (public vs bearer vs WS close codes) with the legacy
  module sections marked as such; README's CRDT-sync bullet (the module is
  unwired) now describes token auth, RBAC and the audit trail.
- The Collaboration Hub is real now, not a demo (Milestone G, G3-02): `c`
  creates and connects a session, `j` edits a session id to join, `Enter`
  connects, `r` retries, `Tab` cycles the server/user/session fields, and
  `Esc` stops editing → disconnects (aborting the worker) → closes; Ctrl+W
  hangs up on its way out too. Fabricated telemetry — the timestamp-derived
  port, "Protocol: WebSocket (TLS)" on a plain connection, `<15ms` latency,
  hardcoded alice/bob/carol members — is gone: the panel shows the real
  server URL, an honest `ws (no TLS)`/`wss (TLS)` transport line, the live
  member list with role badges, connected-for time, and any worker error
  verbatim. The help overlay lists exactly the keys the hub handles.
- `xencode server` is local-first by default (Milestone G, G2-01): it binds
  `127.0.0.1` unless told otherwise (`--host` accepts an IP or `localhost` —
  no DNS resolution behind the user's back), serves TLS only when given
  `--cert` + `--key` (rustls; half a pair is an error), and refuses any
  non-loopback plain bind unless `--allow-insecure-public` is set — with a
  clear-text warning even then. The startup banner prints the honest scheme
  (`ws://` never claims `wss://`) and the audit target; `--audit-path` picks
  the JSONL file (default `~/.xencode/audit.jsonl`, `none` disables), so the
  G1-04 sink is now actually wired. The banner's stale `/ws/{session_id}/{username}`
  route text was corrected to the real `/ws/{session_id}`.
- The WebSocket is no longer an identity claim (Milestone G, G1-03): the route
  is `/ws/{session_id}` — the username is gone from the URL — and the first
  frame must be `auth` with a token the server issued (5 s timeout). Joins run
  through `WorkspaceManager::join` (self-join lands as Editor; the session
  creator keeps Admin), `activity` relay requires Editor+ — a Viewer gets an
  `rbac_denied` error frame and a `Denied` audit entry — and the relayed
  `user` is always the server's authenticated identity, never whatever the
  client's JSON claims. `MAX_SESSION_MEMBERS` (10) is enforced with a 4409
  close (existing members may still reconnect); rejections arrive as an error
  frame plus a 44xx close (4401 bad token, 4404 no session). The wire format
  lives once in `xencode-collaboration-rs::wire` (serde `type`-tagged enums
  shared by server and future client), the server's parallel in-memory
  session map is deleted, and presence (who is connected) is now distinct
  from membership (who belongs). Covered by 12 end-to-end handshake tests
  over in-process duplex pipes — no ports.
- Server auth is real (Milestone G, G1-02): `xencode-server-rs` gained a
  `TokenStore` (random `xencode_<uuid v4 hex>` tokens, 24 h TTL, pruned on
  issue and lookup, constant-time compare) and an `Authed` bearer extractor.
  `/auth/login` now rejects the previously-silently-ignored `api_key` with a
  400 instead of minting a token anyway, `/auth/verify` resolves a token it
  actually issued — the old prefix-and-length check that accepted any >20-char
  `xencode_` string is gone. `POST /sessions/create`, `GET /sessions/{id}`
  (which now 404s for unknown ids instead of an empty 200) and the llama.cpp
  load/unload endpoints require a valid token; permissive CORS was deleted;
  `/api/llamacpp/status` and the public `/api/config` no longer leak the
  model path, executable or launch args
- Collaboration RBAC groundwork (Milestone G, G1-01): the sole admin can no longer
  be *demoted* by an admin role-change (only removal was guarded), both guards now
  write `Denied` audit events, and `WorkspaceManager` gained `join` (idempotent
  self-join as Editor), `create_workspace_with_id` and `log_denied` — the API the
  hardened server will consume
- Docs close-out (Milestone F4): NEXT_PLAN.md marks Milestone F complete and its
  backlog now carries only team-mode hardening; the TUI panel count (24),
  Feature Navigator entries (17), key tables, `xencode advise` docs and the
  CLI subcommand list were re-verified against the tree — 13 crates, 508 tests,
  zero warnings at this commit
- Docs close-out (Milestone D4): NEXT_PLAN.md marks Milestone D complete and drops stale backlog claims (the real-time watcher, multimodal inputs and PR review have all shipped); NEXT_PLAN_TASKS.md subcommand checklist refreshed from `xencode --help`; counts re-verified against a full run — 13 crates, 494 tests, zero warnings
- Docs close-out (Milestone E7): QUICK_START/USER_MANUAL TUI sections rewritten against the real keymap — removed the never-existing `/help` `/models` `/model` `/clear` `/exit` slash commands, documented the help overlay, multiline input, history recall and Tab completion; README panel count corrected to 21; the unbound `h:refresh` Provider Health status hint now shows the working `Ctrl+H`
- TUI key handling restructured into a new `keymap.rs`: global Ctrl chord table + per-focus handlers replace the ~900-line `match` in the event loop. Panels now own their letter keys — SecurityAuditor `s` (sort toggle) and CustomModels `s` (save profile) work again instead of opening Settings — and `j`/`k` are typable in every text field (commit message, ByteBot command, settings URL edit); ByteBot history recall is ↑ only (E6-01)
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