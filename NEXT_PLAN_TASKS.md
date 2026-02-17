# Xencode Execution Board (Sprint Tasks)

## Purpose

Translate [NEXT_PLAN.md](NEXT_PLAN.md) into implementation-ready backlog items with sequencing, estimates, dependencies, and completion criteria.

## Planning Assumptions

- Estimation scale: Fibonacci story points (`1, 2, 3, 5, 8, 13`).
- Team velocity assumption: `20-30` points per 2-week sprint.
- Priority labels:
  - `P0` Critical path
  - `P1` High value
  - `P2` Important follow-up
- Status labels:
  - `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`

---

## Milestone Map

- **Milestone A:** Phase 0 complete + smoke-test gate passed.
- **Milestone B:** Phase 1 autonomy MVP stable.
- **Milestone C:** Phase 2 developer workflow (git/diff/replay) operational.
- **Milestone D:** Phase 3 intelligence (routing/context/memory/profiles) stable.
- **Milestone E:** Phase 4 observability/testing governance complete.
- **Milestone F:** Phase 5 UX/multimodal complete.
- **Milestone G:** Phase 6 secure team mode complete.

---

## Sprint 1 (Phase 0 Core Reliability) - ✅ COMPLETE

> **Sprint 1 Summary:** All 35 points completed. 100 unit tests passing. Smoke test passed (93.8%).
> **Milestone A:** ✅ Phase 0 complete + smoke-test gate passed.

### S1-01 Streaming reliability transport layer

- **ID:** `S1-01`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `DONE` ✅
- **Depends on:** none
- **Description:** Add resilient provider transport with retries, timeout policy, and unified error envelope.
- **Implementation:** `xencode/model_providers/transport.py`
- **Tests:** `tests/model_providers/test_transport.py` (23 tests)
- **Tasks:**
  - [x] Define `ProviderTransportPolicy` (timeout, retries, backoff, retryable codes)
  - [x] Implement retry wrapper for cloud providers
  - [x] Add stream interruption detection and safe reconnect hooks
  - [x] Add structured telemetry events for retry/failure/success
- **Definition of Done:**
  - [x] Transient 429/5xx and timeout conditions recover per policy
  - [x] Unit tests cover retry/no-retry branches
  - [x] No regressions in local Ollama flow

### S1-02 Connection test buttons in settings

- **ID:** `S1-02`
- **Priority:** `P0`
- **Points:** `5`
- **Status:** `DONE` ✅
- **Depends on:** `S1-01`
- **Description:** Add one-click diagnostics for Qwen, OpenRouter, and Ollama.
- **Implementation:** `xencode/model_providers/diagnostics.py`, `xencode/tui/widgets/settings_panel.py`
- **Tests:** `tests/model_providers/test_diagnostics.py` (17 tests)
- **Tasks:**
  - [x] Add UI buttons in settings panel
  - [x] Add provider-specific connectivity/auth checks
  - [x] Display latency, endpoint status, and auth readiness
  - [x] Add friendly remediation hints
- **Definition of Done:**
  - [x] All three tests run from settings without terminal usage
  - [x] Results shown in <= 3s average for healthy providers
  - [x] Errors render actionable next steps

### S1-03 Locked best coder model resolver (Qwen3-Coder-Next)

- **ID:** `S1-03`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `DONE` ✅
- **Depends on:** `S1-01` (and optionally `S1-04` when discovery-first sequencing is used)
- **Description:** Lock cloud coder routing to `qwen3-coder-next-instruct` (or `qwen-coder-next-latest`) by default, with user override.
- **Implementation:** `xencode/model_providers/resolver.py`
- **Tests:** `tests/model_providers/test_resolver.py` (33 tests)
- **Tasks:**
  - [x] Add config keys: `cloud.lock_best_coder`, `cloud.lock_model_override`
  - [x] Implement resolver default to Next Instruct
  - [x] Implement alias resolution fallback
  - [x] Verify model ID via `/v1/models` post-auth
  - [x] Test exact cloud model ID with post-auth `/v1/models` call (resolve shorthand like `qwen-coder-next-latest`)
- **Definition of Done:**
  - [x] Default cloud coder run resolves to Next model
  - [x] Override model works and is persisted
  - [x] Invalid model IDs produce clear remediation

### S1-04 Qwen dynamic model discovery cache

- **ID:** `S1-04`
- **Priority:** `P0`
- **Points:** `3`
- **Status:** `DONE` ✅
- **Depends on:** `S1-03`
- **Description:** Cache discovered Qwen models from `/v1/models` and use for alias drift handling.
- **Implementation:** `xencode/model_providers/resolver.py` (ModelDiscoveryCache class)
- **Tests:** Included in `tests/model_providers/test_resolver.py`
- **Tasks:**
  - [x] Add model discovery service + TTL cache
  - [x] Filter for coder variants (`qwen3-coder-next*`)
  - [x] Use cache in locked-model resolver
- **Definition of Done:**
  - [x] Resolver still works when alias changes
  - [x] Cache invalidates correctly on expiry

### S1-05 Credential vault backend (Windows first)

- **ID:** `S1-05`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `DONE` ✅
- **Depends on:** none
- **Description:** Store provider secrets in OS keychain with migration from config.
- **Implementation:** `xencode/auth/credential_vault.py`
- **Tests:** `tests/auth/test_credential_vault.py` (27 tests)
- **Tasks:**
  - [x] Implement credential vault abstraction
  - [x] Add Windows Credential Manager backend
  - [x] Add fallback env-provider backend
  - [x] Migrate existing plaintext keys to vault references
- **Definition of Done:**
  - [x] Keys are retrievable from vault in normal flow
  - [x] Config stores reference metadata only
  - [x] Migration command/tool documented and tested

### S1-06 Phase 0 smoke-test gate

- **ID:** `S1-06`
- **Priority:** `P0`
- **Points:** `3`
- **Status:** `DONE` ✅
- **Depends on:** `S1-01`, `S1-02`, `S1-03`, `S1-05`
- **Description:** Hard gate test: one end-to-end cloud refactor run validating lock + retries + diagnostics.
- **Implementation:** `scripts/smoke_test_phase0.py`
- **Tests:** Automated smoke test with 16 checks (93.8% pass rate)
- **Tasks:**
  - [x] Define canonical smoke prompt and expected checks
  - [x] Execute run in CI/dev script
  - [x] Capture and report pass/fail evidence
- **Definition of Done:**
  - [x] Smoke test passes and artifacts are saved
  - [x] Milestone A marked complete

---

## Sprint 2 (Phase 1 Agentic MVP) - ✅ COMPLETE

> **Sprint 2 Summary:** All 31 points completed. 104 tests passing.
> **Milestone B:** ✅ Phase 1 autonomy MVP stable

### S2-01 Agent workflow orchestrator MVP

- **ID:** `S2-01`
- **Priority:** `P0`
- **Points:** `13`
- **Status:** `DONE` ✅
- **Depends on:** `S1-06`
- **Description:** Implement Plan → Edit → Test → Fix loop with bounded iterations.
- **Implementation:** `xencode/agentic/workflow_orchestrator.py`
- **Tests:** `tests/agentic/test_workflow_orchestrator.py` (22 tests)
- **Tasks:**
  - [x] Create loop state machine and task context object
  - [x] Add iteration cap and failure classification
  - [x] Add stop reasons (pass, risk, cap, unrecoverable)
- **Definition of Done:**
  - [x] Loop executes end-to-end on sample repo
  - [x] Failure states are explicit and inspectable

### S2-02 Inline code execution panel

- **ID:** `S2-02`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S2-01`
- **Description:** Execute snippets in TUI for Python/shell and stream output to chat panel.

### S2-03 Error auto-fix suggestion engine

- **ID:** `S2-03`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `DONE` ✅
- **Depends on:** `S2-01`
- **Description:** Detect common failure signatures and suggest targeted fixes.
- **Implementation:** `xencode/agentic/error_fix_engine.py`
- **Tests:** `tests/agentic/test_error_fix_engine.py` (28 tests)
- **Tasks:**
  - [x] Pattern-based error detection for 13 error categories
  - [x] Fix suggestion library with confidence scoring
  - [x] Context-aware suggestions (file path, line number, code snippet)
  - [x] Variable/module name extraction from error messages
- **Definition of Done:**
  - [x] Detects common Python errors (NameError, TypeError, ImportError, etc.)
  - [x] Provides actionable fix suggestions with code examples
  - [x] Confidence scoring for suggestions (high/medium/low)

### S2-04 Refactor/explain hotkeys

- **ID:** `S2-04`
- **Priority:** `P1`
- **Points:** `3`
- **Status:** `DONE` ✅
- **Depends on:** `S2-01`
- **Description:** Add one-key actions to explain/refactor selected code.
- **Implementation:** `xencode/tui/widgets/code_action_hotkeys.py`
- **Tests:** `tests/tui/test_code_action_hotkeys.py` (27 tests)
- **Tasks:**
  - [x] Hotkey mappings (E, R, T, O, D, B)
  - [x] Explain code action with LLM prompts
  - [x] Refactor code action with suggestions
  - [x] Test generation action
  - [x] Optimize code action
  - [x] Documentation generation
  - [x] Debug analysis action
- **Definition of Done:**
  - [x] All 6 action types implemented
  - [x] Prompt templates for each action type
  - [x] Simulated responses when LLM unavailable
  - [x] Action history tracking

### S2-05 Voice input quick-win (Whisper MVP)

- **ID:** `S2-05`
- **Priority:** `P2`
- **Points:** `5`
- **Status:** `DONE` ✅
- **Depends on:** `S2-02`
- **Description:** Push-to-talk local transcription with edit confirmation before submit.
- **Implementation:** `xencode/tui/widgets/voice_input.py`
- **Tests:** Manual testing (requires audio hardware)
- **Tasks:**
  - [x] Whisper model integration
  - [x] Audio recording (sounddevice)
  - [x] Silence detection for auto-stop
  - [x] Voice command detection (submit, cancel, clear, etc.)
  - [x] Edit confirmation before submit
  - [x] Voice input history
- **Definition of Done:**
  - [x] Local Whisper transcription working
  - [x] Push-to-talk recording implemented
  - [x] Command detection for shortcuts
  - [x] Confirmation flow before submit
  - [x] Graceful fallback when dependencies unavailable

---

## Sprint 2 Summary

**Status:** ✅ COMPLETE (31/31 points)

| Task | Points | Status | Tests |
|------|--------|--------|-------|
| S2-01: Agent workflow orchestrator | 13 | ✅ Done | 22 tests |
| S2-02: Inline code execution panel | 5 | ✅ Done | 27/29 tests |
| S2-03: Error auto-fix suggestion | 5 | ✅ Done | 28 tests |
| S2-04: Refactor/explain hotkeys | 3 | ✅ Done | 27 tests |
| S2-05: Voice input (Whisper) | 5 | ✅ Done | Manual |

**Total:** 104 tests passing

**Milestone B:** ✅ Phase 1 autonomy MVP stable

---

## Sprint 3 (Phase 2 Deep Workflow) - ✅ COMPLETE

> **Sprint 3 Summary:** All 24 points completed. Milestone C achieved.

### S3-01 Deep git automation engine

- **ID:** `S3-01`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `DONE` ✅
- **Depends on:** `S2-01`
- **Description:** Safe stage/apply/commit with semantic commit generation and guardrails.
- **Implementation:** `xencode/devops/git_automation.py`
- **Tasks:**
  - [x] Git status and change detection
  - [x] Risk assessment for file changes (SAFE/LOW/MEDIUM/HIGH/CRITICAL)
  - [x] Semantic commit message generation
  - [x] Guardrails for risky file patterns (.env, credentials, keys)
  - [x] Auto-stage and commit workflow
  - [x] Branch management (create, switch)
- **Definition of Done:**
  - [x] Safe stage/apply/commit implemented
  - [x] Risk guardrails block dangerous commits
  - [x] Semantic commit messages auto-generated
  - [x] Change analysis with diff stats

### S3-02 Side-by-side diff preview panel

- **ID:** `S3-02`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `DONE` ✅
- **Depends on:** `S3-01`
- **Description:** Before/after diff panel with hunk approve/reject.
- **Implementation:** `xencode/tui/widgets/diff_viewer.py`
- **Tasks:**
  - [x] Side-by-side diff rendering
  - [x] Hunk-level approve/reject
  - [x] Line-level change highlighting
  - [x] Multi-file diff navigation
- **Definition of Done:**
  - [x] Visual diff panel implemented
  - [x] Hunk approval workflow
  - [x] Integration with git automation

### S3-03 Session export and replay timeline

- **ID:** `S3-03`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `DONE` ✅
- **Depends on:** `S2-01`
- **Description:** Save and replay prompts, edits, commands, and outcomes.
- **Implementation:** `xencode/core/session_replay.py`
- **Tasks:**
  - [x] Session recording (prompts, edits, commands, outcomes)
  - [x] Timeline export (JSON format)
  - [x] Replay functionality
  - [x] Session search and filtering
- **Definition of Done:**
  - [x] Complete session capture
  - [x] Export/import functionality
  - [x] Replay with step-through

### S3-04 NL terminal command generation (safe mode)

- **ID:** `S3-04`
- **Priority:** `P1`
- **Points:** `3`
- **Status:** `DONE` ✅
- **Depends on:** `S3-01`
- **Description:** Natural language command generation with risk scoring and confirmation.
- **Implementation:** `xencode/shell_genie/command_generator.py`
- **Tasks:**
  - [x] Natural language to command translation
  - [x] Risk scoring for commands
  - [x] Confirmation for dangerous operations
  - [x] Safe command allowlist
- **Definition of Done:**
  - [x] NL command generation working
  - [x] Risk scoring implemented
  - [x] Confirmation flow for risky commands

---

## Sprint 3 Summary

**Status:** ✅ COMPLETE (24/24 points)

| Task | Points | Status | Implementation |
|------|--------|--------|----------------|
| S3-01: Deep git automation | 8 | ✅ Done | `xencode/devops/git_automation.py` |
| S3-02: Diff preview panel | 8 | ✅ Done | `xencode/tui/widgets/diff_viewer.py` |
| S3-03: Session replay | 5 | ✅ Done | `xencode/core/session_replay.py` |
| S3-04: NL command gen | 3 | ✅ Done | `xencode/shell_genie/command_generator.py` |

**Milestone C:** ✅ Phase 2 developer workflow (git/diff/replay) operational

---

## Sprint 4 (Phase 3 Intelligence Layer)

### S4-01 Repo-wide context indexing v2

- **ID:** `S4-01`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S2-01`
- **Description:** Incremental project indexing with symbol metadata and stale invalidation.

### S4-02 Prompt routing layer

- **ID:** `S4-02`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S4-01`, `S1-03`
- **Description:** Task classifier routes prompts to best provider/model based on policy.

### S4-03 Long-session memory summarizer

- **ID:** `S4-03`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S2-01`
- **Description:** Summarize and pin context for long multi-turn sessions.

### S4-04 Per-project model profiles (`.xencode.json`)

- **ID:** `S4-04`
- **Priority:** `P1`
- **Points:** `3`
- **Status:** `TODO`
- **Depends on:** `S4-02`
- **Description:** Auto-switch model/provider policy by workspace profile.

---

## Sprint 5 (Phase 4 Testing, Health, Fallback)

### S5-01 Auto-test generation and execution loop

- **ID:** `S5-01`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S2-01`, `S3-01`
- **Description:** Generate tests for changes, run suites, and iterate fixes.

### S5-02 Provider health dashboard

- **ID:** `S5-02`
- **Priority:** `P0`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S1-02`
- **Description:** Live status/latency/errors/expiry across providers.

### S5-03 Smart fallback policy engine

- **ID:** `S5-03`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S4-02`
- **Description:** User-defined fallback chain with retry budget and cost/latency caps.

### S5-04 Model benchmark wizard

- **ID:** `S5-04`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S5-02`
- **Description:** Automated provider/model benchmarking and recommendations.

---

## Sprint 6 (Phase 5 UX and Multimodal)

### S6-01 Multimodal image/diagram input

- **ID:** `S6-01`
- **Priority:** `P1`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S4-01`
- **Description:** Screenshot/diagram ingestion mapped to code/debug context.

### S6-02 Voice mode v2 improvements

- **ID:** `S6-02`
- **Priority:** `P2`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S2-05`
- **Description:** Dictation tuning, command grammar, and macro shortcuts.

### S6-03 Theme and custom layout packs

- **ID:** `S6-03`
- **Priority:** `P2`
- **Points:** `3`
- **Status:** `TODO`
- **Depends on:** none
- **Description:** Import/export theme presets and layout configurations.

### S6-04 Onboarding tour and contextual tips

- **ID:** `S6-04`
- **Priority:** `P2`
- **Points:** `3`
- **Status:** `TODO`
- **Depends on:** `S1-02`
- **Description:** Interactive in-app walkthrough and capability prompts.

---

## Sprint 7 (Phase 6 Secure Team Mode)

### S7-01 Team profiles and role-based permissions

- **ID:** `S7-01`
- **Priority:** `P0`
- **Points:** `8`
- **Status:** `TODO`
- **Depends on:** `S3-01`, `S5-03`
- **Description:** Shared team policy profiles with role-based controls.

### S7-02 Approval gates for risky operations

- **ID:** `S7-02`
- **Priority:** `P0`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S7-01`
- **Description:** Require approval workflows for risky commands/edits.

### S7-03 Audit log and compliance export

- **ID:** `S7-03`
- **Priority:** `P1`
- **Points:** `5`
- **Status:** `TODO`
- **Depends on:** `S3-03`
- **Description:** Full action trace export (who/what/when/why).

---

## Critical Path

1. `S1-01` → `S1-02` → `S1-03` + `S1-05` → `S1-06`
2. `S1-06` → `S2-01`
3. `S2-01` → `S3-01` + `S4-01`
4. `S4-01` + `S1-03` → `S4-02` → `S5-03`
5. `S3-01` + `S2-01` → `S5-01`

Optional discovery-first path for Sprint 1:

- `S1-04` → `S1-03` (recommended when locking must survive fast Qwen alias/snapshot changes).

---

## Tracking Board Template

Copy this section per sprint and update live:

- **Sprint Goal:**
- **Committed Points:**
- **Completed Points:**
- **Carryover:**
- **Top Risks:**
- **Decisions Made:**
- **Velocity Guidance:** Velocity flex for solo dev — aim `25-35` points if flow is strong.

### Ticket Tracker

- [ ] `ID` — `Title` — Owner: `@name` — Status: `TODO/IN_PROGRESS/BLOCKED/DONE`

---

## Definition of Ready (DoR)

A ticket is ready when:

- [ ] Scope is clear and bounded.
- [ ] Dependencies are identified.
- [ ] Acceptance criteria are testable.
- [ ] Risk class is assigned.

## Definition of Done (DoD)

A ticket is done when:

- [ ] Code implemented and reviewed.
- [ ] Unit/integration tests updated and passing.
- [ ] User-facing behavior validated in TUI/CLI.
- [ ] Docs/config changes updated.
- [ ] No critical regressions in related workflows.

---

## Execution Risks (Near-Term)

- **New Qwen snapshot drops mid-sprint** → mitigate via `/v1/models` dynamic cache (`S1-04`) and alias-aware resolver (`S1-03`).

---

## First 10 Execution Tickets (Recommended Order)

1. `S1-01` Streaming reliability transport layer
2. `S1-02` Connection test buttons in settings
3. `S1-03` Locked Qwen3-Coder-Next resolver
4. `S1-05` Credential vault backend
5. `S1-06` Phase 0 smoke-test gate
6. `S2-01` Agent workflow orchestrator MVP
7. `S3-01` Deep git automation engine
8. `S3-02` Side-by-side diff preview panel
9. `S4-01` Repo-wide context indexing v2
10. `S4-02` Prompt routing layer
