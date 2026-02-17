# Xencode Next Implementation Plan

## Objective

Deliver a production-grade autonomous coding assistant platform by expanding Xencode across agent autonomy, cloud/local reliability, developer workflow acceleration, and enterprise-grade security/team capabilities.

## Planning Principles

- Build in safe increments with feature flags and rollback paths.
- Ship an MVP for each pillar first, then optimize UX and performance.
- Keep local-first reliability while adding robust cloud pathways.
- Validate each phase with measurable acceptance criteria.

---

## Phase 0 (Weeks 1-2): Foundation Hardening

### Phase 0 Features

1. Streaming Reliability (retry/resume on blips)
2. Connection Test Buttons (per-provider diagnostics)
3. Locked Best Coder Model (force Qwen3-Coder-Next Instruct for cloud)
4. Credential Vault (OS keychain for secrets)

### Phase 0 Implementation Plan

- Add a provider transport layer with:
  - transient retry policy (timeouts, 429, 5xx)
  - resumable stream abstraction (resume token/chunk index where available)
  - unified structured error model
- Extend settings panel with test actions:
  - Qwen test
  - OpenRouter test
  - Ollama local test
  - return latency, auth state, endpoint health
- Add cloud model policy config:
  - `cloud.lock_best_coder=true`
  - resolver defaults to `qwen3-coder-next-instruct` (alias: `qwen-coder-next-latest`)
  - allow user override (`cloud.lock_model_override`) for alternate pinned models
  - fallback only if policy allows
  - after auth, verify exact model ID against `https://chat.qwen.ai/v1/models` (quick curl/API check)
- Add model discovery behavior for Qwen cloud:
  - call `/v1/models` and cache available Qwen coder variants
  - if Next alias changes, resolve to newest matching `qwen3-coder-next*` instruct variant
  - keep hard default lock on Next unless user override is set
- Implement secure credential backend:
  - Windows Credential Manager primary backend
  - environment variable fallback
  - migration from plaintext config to vault references

### Phase 0 Deliverables

- Robust stream layer used by all cloud providers.
- Working provider diagnostics from TUI settings.
- Locked coder model resolver with override controls.
- Qwen model discovery and ID verification via `/v1/models` endpoint.
- Vault-backed secret storage and migration utility.

### Phase 0 Acceptance Criteria

- 99%+ successful completion on transient network faults with retries.
- Provider test buttons return status in less than 3 seconds average.
- No plaintext API keys required in config for normal usage.
- Locked model policy enforces cloud coder selection consistently.
- Locked target resolves to Qwen3-Coder-Next Instruct (or configured override) in cloud runs.

### Phase 0 to 1 Gate (Smoke Test Milestone)

- Run one end-to-end cloud coder smoke test before Phase 1 autonomy work:
  - prompt type: simple refactor task
  - validates locked model selection + streaming retries + diagnostics visibility
  - must pass to unblock Agentic MVP implementation

---

## Phase 1 (Weeks 3-5): Agentic Coding Core

### Phase 1 Features

1. Full Agentic Workflow (Plan → Edit Files → Test → Iterate)
2. Inline Code Execution (run snippets live in TUI)
3. Error Auto-Fix Suggestions
4. Refactor/Explain Hotkeys
5. Voice Input Mode Quick Win (local Whisper, push-to-talk)

### Phase 1 Implementation Plan

- Build `AgentWorkflowOrchestrator` loop states:
  1. Plan task
  2. Select files
  3. Apply edits
  4. Run tests/lint
  5. Classify failures
  6. Repair iteration
  7. Stop when green or max iterations reached
- Add guarded file-edit policy:
  - editable scope constraints
  - dry-run preview mode
  - stop-on-risk operations
- Add inline execution panel:
  - language-aware snippet runner
  - stdout/stderr capture
  - optional output-to-context insertion
- Add diagnostics suggestion engine for top failure patterns:
  - import errors
  - assertion mismatches
  - syntax/type errors
- Add hotkeys for explain/refactor/test generation actions.
- Add quick voice input path early for momentum:
  - local Whisper transcription for prompt entry
  - push-to-talk with confirmation before send
  - route transcript through normal agentic loop

### Phase 1 Deliverables

- End-to-end autonomous fix loop.
- Inline execution in TUI.
- Auto-fix suggestion cards in chat/review panels.
- Productivity hotkeys integrated into TUI help/actions.
- Voice prompt input MVP integrated with chat submit flow.

### Phase 1 Acceptance Criteria

- Agent resolves over 60% benchmark issues without manual edits (initial target).
- Average loop turnaround is under 20 seconds for small repos.
- Inline execution supports Python and shell safely.
- Suggestion engine handles top 10 recurring error categories.
- Voice prompt round-trip works reliably for development prompts.

---

## Phase 2 (Weeks 6-8): Deep Developer Workflow Integration

### Phase 2 Features

1. Deep Git Integration (auto-diff review, smart commits, safe staging)
2. Diff Preview Panel (side-by-side before/after)
3. Session Export/Replay
4. Terminal Command Generation (natural language to shell)

### Phase 2 Implementation Plan

- Git workflow engine:
  - patch-level staging
  - semantic commit message generation
  - commit guardrails (tests/lint checks)
  - rollback checkpoint tags
- Diff preview UI:
  - side-by-side renderer
  - intra-line highlights
  - apply/reject hunk actions
- Session recorder:
  - prompts, model decisions, edits, command outputs
  - resumable replay timeline
- NL terminal generation with safety classifier:
  - command risk scoring
  - mandatory confirmation for risky commands

### Phase 2 Deliverables

- Production-grade git assistant in TUI.
- Reviewable diff workflow before file writes/commits.
- Exportable and replayable agent sessions.
- Safe NL-to-terminal command feature.

### Phase 2 Acceptance Criteria

- Full flow works inside TUI: prompt → edit → diff → stage → commit.
- Replay reconstructs at least 95% of session events accurately.
- High-risk command confirmation is never bypassed.

---

## Phase 3 (Weeks 9-11): Context, Memory, and Routing Intelligence

### Phase 3 Features

1. Repo-Wide Context Indexing
2. Prompt Routing Layer (auto task detection + routing)
3. Multi-Turn Memory Management (smart summarization)
4. Per-Project Model Profiles (`.xencode.json`)

### Phase 3 Implementation Plan

- Upgrade indexing pipeline:
  - incremental indexing
  - language-aware chunking
  - symbol metadata and dependency edges
  - stale index invalidation
- Build routing policy engine:
  - classify tasks (debug/refactor/generate/review/docs)
  - route by latency/cost/quality policy
- Build long-session memory manager:
  - context window budgeting
  - rolling summarization
  - pinned critical constraints
- Add per-project profile support:
  - `.xencode.json` local profile
  - model defaults, provider chain, test command presets

### Phase 3 Deliverables

- High-quality repo context retrieval.
- Automatic prompt-to-model/provider routing.
- Stable long-horizon multi-turn conversations.
- Workspace-specific profile behavior.

### Phase 3 Acceptance Criteria

- Context retrieval precision significantly improves on repo QA tasks.
- Routing chooses expected provider/model class in over 85% audited tasks.
- Memory summarization prevents context overflow in long sessions.

---

## Phase 4 (Weeks 12-13): Testing, Observability, and Fallback Governance

### Phase 4 Features

1. Auto-Test Generation and Execution
2. Model Benchmark Wizard
3. Provider Health Dashboard
4. Smart Fallback Policies (configurable chains + caps)

### Phase 4 Implementation Plan

- Test automation engine:
  - infer framework (pytest/jest/etc.)
  - generate tests for changed code
  - execute targeted and full suites
  - auto-repair cycle for failing tests
- Benchmark wizard:
  - standardized prompt suites
  - score latency, cost, quality
  - recommendations by use-case
- Health dashboard:
  - per-provider status cards
  - token expiry warnings
  - rolling latency/error charts
- Fallback policy DSL:
  - chain ordering
  - cost cap, latency cap, retry budget
  - hard-fail vs soft-fallback mode

### Phase 4 Deliverables

- Automatic test loop integrated with agent workflow.
- Model benchmarking and recommendation UX.
- Real-time provider observability.
- User-defined fallback policy engine.

### Phase 4 Acceptance Criteria

- Auto-test loop runs with one action and reports actionable summary.
- Dashboard updates continuously with accurate provider telemetry.
- Fallback engine enforces caps and chain rules deterministically.

---

## Phase 5 (Weeks 14-16): Multimodal and UX Expansion

### Phase 5 Features

1. Multi-Modal Input (screenshots/diagrams → code/debug)
2. Advanced Voice Mode V2 (dictation quality, command grammar, macros)
3. Theme/Custom Layout Packs
4. Onboarding Tour and Tips

### Phase 5 Implementation Plan

- Add multimodal ingestion:
  - image attach flow
  - OCR/vision parsing
  - map visual issues to code context
- Add voice input:
  - improve dictation quality and terminology adaptation
  - add voice command grammar/macros for repetitive coding actions
  - add post-processing for punctuation and code block formatting
- Add layout/theme packs:
  - panel presets
  - custom keymaps
  - import/export themes
- Expand onboarding:
  - progressive tips
  - capability walkthrough
  - first-task guided completion

### Phase 5 Deliverables

- Vision-assisted coding and debugging flow.
- Voice-driven prompt input.
- Advanced personalization options.
- Faster user ramp-up and feature discovery.

### Phase 5 Acceptance Criteria

- Users can attach screenshot/diagram and receive code-targeted output.
- Voice input reaches acceptable coding-prompt accuracy baseline.
- Onboarding completion correlates with reduced first-week drop-off.

---

## Phase 6 (Weeks 17-19): Secure Team Mode and Governance

### Phase 6 Features

1. Secure Team Mode (shared configs + roles)
2. Governance extensions (RBAC + approvals)
3. Auditability for agent actions and file mutations

### Phase 6 Implementation Plan

- Team workspace profiles:
  - shared provider policies
  - role-based command permissions
  - approval gates for risky actions
- Action audit logs:
  - who/what/when for edits, commands, commits
  - export for compliance
- Team-safe secret references:
  - vault-backed references only
  - no secret value leakage to logs

### Phase 6 Deliverables

- Multi-user secure collaboration mode.
- Governed autonomy controls.
- Compliance-ready audit trail.

### Phase 6 Acceptance Criteria

- RBAC enforcement for restricted actions.
- Full traceability for agent-initiated changes.
- Shared team profile onboarding in minutes.

---

## Feature Mapping Matrix (All Requested Items)

| Requested Feature | Phase |
| --- | --- |
| Full Agentic Workflow | 1 |
| Deep Git Integration | 2 |
| Repo-Wide Context Indexing | 3 |
| Auto-Test Generation & Execution | 4 |
| Provider Health Dashboard | 4 |
| Smart Fallback Policies | 4 |
| Connection Test Buttons | 0 |
| Streaming Reliability | 0 |
| Per-Project Model Profiles | 3 |
| Voice Input Mode | 1 |
| Credential Vault | 0 |
| Locked Best Coder Model | 0 |
| Multi-Modal Input | 5 |
| Prompt Routing Layer | 3 |
| Session Export/Replay | 2 |
| Inline Code Execution | 1 |
| Refactor/Explain Hotkeys | 1 |
| Error Auto-Fix Suggestions | 1 |
| Multi-Turn Memory Management | 3 |
| Diff Preview Panel | 2 |
| Model Benchmark Wizard | 4 |
| Secure Team Mode | 6 |
| Terminal Command Generation | 2 |
| Theme/Custom Layout Packs | 5 |
| Onboarding Tour & Tips | 5 |

---

## Core Technical Additions

### New Modules

- `xencode/agent/workflow_orchestrator.py`
- `xencode/git/automation_engine.py`
- `xencode/context/repo_indexer_v2.py`
- `xencode/testing/auto_test_engine.py`
- `xencode/providers/health_monitor.py`
- `xencode/providers/fallback_policy_engine.py`
- `xencode/security/credential_vault.py`
- `xencode/routing/prompt_router.py`
- `xencode/memory/session_summarizer.py`
- `xencode/session/replay_manager.py`

### Config Extensions

- Provider fallback rules
- Routing preferences
- Per-project model profiles
- Team policy and role bindings
- Stream retry and timeout controls

---

## Milestones and Exit Gates

### Milestone A (after Phase 1)

Autonomous fix loop operational with bounded retries and safety checks.

### Milestone B (after Phase 3)

Project-aware routing + memory + profile switching stable in real repos.

### Milestone C (after Phase 4)

Provider observability, fallback policy, and benchmark wizard production-ready.

### Milestone D (after Phase 6)

Team-grade secure collaboration and auditability complete.

---

## Risks and Mitigations

- Provider API changes → contract tests and compatibility adapters.
- Qwen API evolution cadence (rapid model/version changes) → dynamic model listing via `/v1/models`, alias resolver, and health dashboard integration in Phase 4.
- Autonomous edit regressions → mandatory diff preview and rollback checkpoints.
- Key management complexity → unified vault abstraction with migration tooling.
- Cloud latency spikes → routing caps + fallback policy + health-based selection.
- Large repo indexing cost → incremental index updates and selective embedding.

---

## Recommended Immediate Sprint (Next 2 Weeks)

1. Prioritize Phase 0 reliability first: Streaming Reliability + Connection Test Buttons.
2. Complete remaining Phase 0 items: Locked Qwen3-Coder-Next model policy + Credential Vault.
3. Execute Phase 0→1 smoke-test milestone (one end-to-end cloud refactor run) as hard gate.
4. Start Phase 1 agent loop MVP with test-run/fix iteration.
5. Begin Phase 2 baseline with diff preview and safe git staging.

This sequence gives the fastest path to reliable autonomous coding while keeping cloud/provider behavior safe and observable.
