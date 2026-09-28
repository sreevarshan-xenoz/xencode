# Agent Fabric Research Audit — W01–W04

**Milestone:** T-1  
**Audit date:** 2026-09-27  
**Source:** [Xencode_Next_Research_Pool_3600.md](../Xencode_Next_Research_Pool_3600.md)  
**Status:** complete as a first-pass disposition; new work remains gated by T-2,
T-3, and the Milestone S `AR-1` measurement.

## Scope and method

The 400 source IDs in W01–W04 reduce to 40 distinct themes. Each theme is
repeated ten times with the same generic subsystem labels: inspector, indexer,
validator, planner, simulator, optimizer, policy engine, recovery engine,
analytics view, and export/API. This audit keeps every ID addressable as a
ten-ID range while assessing the underlying theme once. Those variants do not
currently describe ten different user capabilities.

The overlap check used the shipped Rust tree and plan IDs in M, S, and W0–W17.
Vendor evidence combines Milestone S's 2026-09-23 local `--help` matrix with a
read-only refresh of installed CLI versions and help output on 2026-09-27. The
refresh ran each binary's version/help path only; it did not start an agent turn,
use an account, or spend model tokens. Some mise shims could not activate because
the home directory is read-only, so the installed binaries were invoked by their
versioned paths.

| CLI | Version observed | Relevant current help output |
|---|---:|---|
| Codex | 0.157.1 | `exec --json`; `app-server` daemon and schema generation; sandbox choices. |
| Claude Code | 2.1.283 | `--print`, `--output-format stream-json`, session resume, and `--permission-prompts` with `host` or `none`; help says `host` may be the SDK host or a permission-prompt tool. |
| Gemini CLI | 0.61.0 | `--output-format stream-json`; approval modes; `--acp` starts its ACP agent server. Official ACP docs expose `setSessionMode` to change approval level during a session. |
| OpenCode | 1.18.32 | `run --format json`; session resume/fork; `--auto` changes approval behavior. |
| Agy | 1.2.11 | `--output-format stream-json`; session resume; sandbox and execution modes. |
| Crush | 0.96.1 | `run --help` exposed no structured event-output option. This means “not observed in help,” not proof that no other interface exists. |

These help screens establish that several vendors expose machine-readable
streams, not that those streams share an event schema. Their payloads have not
been captured and compared here. The current Gemini help also changes the old
S-0 statement that ACP was not seen for Gemini: Gemini 0.61.0 now documents
`--acp`. That is Gemini acting as an ACP agent/server, which is a different
direction from xencode exposing an ACP server under M-7. Its official ACP
documentation also describes `setSessionMode` as changing approval level during
a session. This is a documented live session-mode seam, but it does not by
itself prove per-request approval interception. Claude's current local help
likewise says `--permission-prompts host|none`; its wording mentions a
permission-prompt tool, but the old standalone `--permission-prompt-tool` flag
is not listed as an option in this version's help. Neither path has been run
end to end here.

Complexity is a rough implementation estimate: S = small, M = several days,
L = a larger multi-week effort. Value and risk are relative judgments, not
measured results. “Existing plan overlap” names the nearest committed or planned
surface; “vendor overlap” names what was actually observed or recorded. No row
promotes a new implementation ID.

## W01 — Agent Interoperability Fabric

| candidate_id / theme | existing_overlap | vendor_overlap | evidence | dependency | xencode_ownership | complexity | value | risk | disposition |
|---|---|---|---|---|---|---|---|---|---|
| X0001–X0010 — protocol adapters | S `AR-4`, `AR-9`; M-5/M-6 | Five observed CLIs offer some headless structured output; the formats and events differ. Crush's run help showed none. | S-0 plus 2026-09-27 CLI help; no stream captures yet. | `AR-1`, then `AR-3`/`AR-9`. | Normalize only the observed cross-vendor contract; vendor-specific session logic stays vendor-owned. | M | H | H | REFINE into `AR-4`/`AR-9`; do not adopt a universal adapter spec before captures. |
| X0011–X0020 — capability cards | S `AR-3`; A2A discovery | A2A defines Agent Cards; Codex, Claude, Gemini and Agy expose CLI flags, subcommands, or agent definitions. | `AR-3` probe and the [A2A v1.0 spec](https://a2a-protocol.org/v1.0.0/). | `AR-1`; define only fields a consumer uses. | Keep xencode's cross-agent capability view; do not mirror a vendor's profile system. | S | M | M | FOLD into `AR-3`; revisit A2A Agent Cards only if xencode interoperates with A2A agents. |
| X0021–X0030 — session translation | S `AR-5`, `AR-7`; W6 `EVd-1` | Vendors already manage their own sessions, resume, fork, and background attachment. | S-0; refreshed Codex, Claude, Gemini, OpenCode and Agy help. | `AR-5` identity and `EVd-1` ledger. | Store stable worker identity and a handoff summary, not a copied vendor transcript. | M | H | H | FOLD into `AR-5`/`AR-7`; vendor session stores remain authoritative. |
| X0031–X0040 — artifact exchange | S `AR-7`, `OR-16`; W17 `OR-4`/`OR-5` | Vendor output can include prose, structured events, and changes in the shared checkout; the portable boundary is not yet measured. | S-5's observed diff/state boundary; `AR-1` still open. | `AR-1`, worktree lease `OR-4`, evidence envelope `OR-16`. | Build a neutral, evidence-backed handoff across vendors. | M | H | H | REFINE into `AR-7`/`OR-16`; test artifact scope after `AR-1`. |
| X0041–X0050 — event normalization | W1 `WF-1`, `EV-2`; S `AR-4`, `AR-9` | Codex JSONL, Claude/Gemini/Agy stream-json, and OpenCode JSON are distinct vendor streams; Crush has no format in the probed help. | Direct help output on 2026-09-27; no captured stream comparison. | `AR-1` captures, then `AR-9` event schema. | Own the common record only where it preserves source facts and provenance. | M | H | H | REFINE into `AR-9`; the earlier S-1 wording “one event schema” is not yet evidenced. |
| X0051–X0060 — version negotiation | S `AR-3`, `AR-9`; M-5/M-6 | MCP and A2A version their protocols; vendor CLI versions evolve independently. | [MCP 2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/) and [A2A v1.0](https://a2a-protocol.org/v1.0.0/); installed versions above. | Select protocol bindings after T-2 and `AR-1`. | Track adapter compatibility and fail clearly on unsupported vendor output. | M | M | M | REFINE into a compatibility field in `AR-3`; avoid a new negotiation protocol. |
| X0061–X0070 — transport bridges | M-5 MCP server, M-6 MCP client, M-7 ACP; S `AR-1` | Codex App Server is bidirectional JSON-RPC; Gemini 0.61.0 serves ACP and documents client-driven approval-level changes; the CLIs also expose stdio or local server modes. | Refreshed local help; [OpenAI App Server article](https://openai.com/index/unlocking-the-codex-harness/); [Gemini ACP docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/acp-mode.md). | T-2 direction matrix; `AR-1`; permission model before remote or write-capable bridges. | Integrate at the boundary that lets independent agents exchange work. | L | H | H | REFINE against M-5/M-6/M-7 and `AR-1`; explicitly distinguish client/server direction. |
| X0071–X0080 — agent discovery | S `AR-2`, `AR-8`; W11 `DB-6` | Codex, Claude, Gemini, OpenCode and Agy have version/help surfaces; the project already probed PATH and mise-managed installs. | S-0 discovery plus 2026-09-27 direct binary versions. | `AR-2` and `AR-8`. | Discover installed workers and report their state; do not install or authenticate them. | S | M | M | FOLD into `AR-2`/`AR-8`; do not add a second discovery service. |
| X0081–X0090 — compatibility testing | S `AR-1`, `AR-3`, `AR-9` | Each vendor's version and output contract can change; no shared conformance suite has been measured. | Version changes since S-0 and the differing help formats are concrete drift evidence. | Capture real output with `AR-1`; define assertions from `AR-9`. | Keep a small local compatibility suite for xencode's normalized contract. | M | H | M | REFINE as `AR-1`/`AR-3` regression cases; defer a broad TCK until real failures justify it. |
| X0091–X0100 — interoperability diagnostics | S `AR-3`, `AR-8`; W11 `DB-6` | Vendor doctor/debug commands exist unevenly; current binary/version/help checks already identify installed and discoverable states. | S-0 plus `AR-8` scope; no evidence yet for a separate diagnostic product. | `AR-2`, `AR-3`, `AR-8`. | Explain why a worker is unavailable or incompatible. | S | L | L | FOLD into `AR-8` and `DB-6`; no standalone diagnostics surface. |

## W02 — Agent Identity & Authority

| candidate_id / theme | existing_overlap | vendor_overlap | evidence | dependency | xencode_ownership | complexity | value | risk | disposition |
|---|---|---|---|---|---|---|---|---|---|
| X0101–X0110 — agent identities | S `AR-5`, `AR-6`; W1 `CX-2` | Vendors identify their own sessions and accounts; they do not provide one cross-vendor worker identity. | S-0 session fields; `CX-2` metrics schema; refreshed session flags. | `AR-5` before `EVd-1`/`CX-2` worker attribution. | Assign a stable local worker ID without claiming it proves the vendor's human/account identity. | S | H | H | FOLD into `AR-5`; document identity's limits. |
| X0111–X0120 — delegation chains | S `OR-3`, `OR-15`, `OR-17`; W7 authority items | Vendor subagents exist (notably Claude custom agents and OpenCode agents); cross-vendor authority does not. | S-0; `OR-15` task contract and `OR-17` veto. | `CAP-1`, `SE-4`, task contract and audit trail. | Preserve who authorized each worker and which task scope it received. | M | H | H | REFINE into `OR-3`/`OR-15`; no separate delegation graph until execution requires it. |
| X0121–X0130 — authority scopes | W7 `CAP-1`, `SE-4`; S `OR-3` | Codex, Claude, Gemini and Agy expose launch-time modes. Gemini ACP documents `setSessionMode` for changing approval level during a session; Claude 2.1.283 help documents `--permission-prompts` with `host` or `none`. Neither live path has been exercised. | S-3/S-13; refreshed CLI help; [Gemini ACP docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/acp-mode.md); [Claude CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage). | `AR-1`, `M-5`, `CAP-1`, `SE-4`. | Set a least-authority task boundary and label configured-at-launch, live session-mode control, and per-request approval separately. | M | H | H | FOLD into `CAP-1`/`OR-3`; update S-13's vendor-exclusive claim after `AR-1` confirms exact runtime behavior. |
| X0131–X0140 — credential brokerage | Existing config/key handling; S-8 rejects taking over vendor auth | Vendors own their sign-in, API keys, and auth stores. | S-0 auth surface; vendor docs and local help; no user request for xencode to manage credentials. | None justifies taking custody; would require a new secrets threat model. | No unique ownership case: credential transfer would widen exposure and duplicate vendors. | L | L | H | REJECT credential brokerage; keep vendor authentication vendor-owned. |
| X0141–X0150 — non-repudiation | W1 `EV-11`; W6 `EVd-1`, `EVd-7`; S `AR-5` | Vendor session logs are vendor-local; they do not attest that xencode observed an action unless captured by xencode. | EV-11 chain verification and documented truncation limits; EVd ledger plan. | `AR-4` observed events plus `EVd-1`; signatures require an external key/identity authority. | Preserve observed events and evidence; do not claim cryptographic proof of vendor intent. | M | M | H | FOLD into `EV-11`/`EVd-1`; keep the external trust anchor limitation explicit. |
| X0151–X0160 — identity rotation | No distinct roadmap item; vendor accounts remain vendor-owned | Vendor credential rotation belongs to each vendor's auth mechanism. | S-0 auth ownership; no xencode-owned credential lifecycle. | A real xencode-issued credential would be prerequisite, but none is planned for workers. | No current xencode ownership or evidence that rotation belongs in this product. | M | L | H | REJECT for external vendor accounts; revisit only if xencode issues worker credentials. |
| X0161–X0170 — trust relationships | W7 trust/security items; S `OR-3`, `OR-17` | Trust is mediated by vendor policy and local project authorization; no cross-vendor trust graph exists in the observed interfaces. | S-3/S-13; W7 dependencies. | `CAP-1`, `SE-4`, observed worker identity, and a human gate. | Make trust decisions for sharing state and accepting evidence across workers. | M | H | H | REFINE into W7 and `OR-3`/`OR-17`; do not create reputation scores. |
| X0171–X0180 — impersonation defense | S `AR-5`; W7 identity/auth controls | Local process identity and vendor login are distinct; CLI flags alone cannot prove who is behind a logged-in account. | S-0 and local version probes; no adversarial impersonation test has been run. | `AR-1`, OS process ownership, and a defined credential boundary. | Protect worker-to-task attribution inside xencode, not vendor account identity. | M | M | H | RESEARCH MORE: define a concrete spoofing threat and test before proposing controls. |
| X0181–X0190 — authorization proofs | W7 `CAP-1`, `SE-4`; S `OR-3`; W1 `EV-11` | MCP/A2A have protocol authorization mechanisms; local CLIs mostly expose startup policy and, for Claude, an approval callback. | MCP 2026-07-28 auth changes; A2A v1.0; local S-13 seam evidence. | `AR-1`, M-5, capability policy, audit. | Prove which xencode grant was checked; don't imply a protocol token proves user intent. | L | M | H | REFINE into grant/audit records under `OR-3`; “proof” needs a narrower claim. |
| X0191–X0200 — identity diagnostics | S `AR-8`; W11 `DB-6` | Vendor doctor commands are uneven; auth state may be private or not machine-readable. | S-0 plus refreshed help; no auth files inspected. | `AR-2`/`AR-8`; honor vendor auth boundaries. | Report installed/responsive and only authentication signals explicitly exposed. | S | M | M | FOLD into `AR-8`/`DB-6`; never parse private credential stores. |

## W03 — Agent Observability

| candidate_id / theme | existing_overlap | vendor_overlap | evidence | dependency | xencode_ownership | complexity | value | risk | disposition |
|---|---|---|---|---|---|---|---|---|---|
| X0201–X0210 — distributed traces | W1 `EV-2`, `EV-11`; W6 `EVd-1`; S `AR-4` | Codex App Server emits bidirectional events; other CLIs expose structured output with vendor-specific shapes. | `turns.jsonl`, EV-11, and current help matrix. | `AR-9` normalized events and `EVd-1` run ledger. | Correlate work across vendor processes and xencode decisions. | L | H | H | REFINE into `EVd-1`/`AR-4`; defer an OTel exporter until the event data is real. |
| X0211–X0220 — agent spans | W1 `EV-2`; S `AR-4`, `AR-6` | Vendor streams have step/turn events; boundaries differ and may omit internal work. | Direct CLI help only; no stream capture or semantic span comparison. | `AR-1`, then `AR-4`/`AR-9`. | Mark spans xencode observed and keep vendor-reported phases labeled as such. | M | M | M | FOLD into `AR-4` and EV-2 trace extension; no separate span engine. |
| X0221–X0230 — tool telemetry | W1 `EV-2`, `CX-1`; S `AR-4`, `AR-6` | Vendor JSON output can report tool events unevenly; Claude has a permission callback, others show launch policy. | EV-2 stores tool-call outcomes; refreshed output-format help; `AR-1` pending. | Event capture, redaction, worker identity. | Attribute tool activity and permission mode to each worker. | M | H | H | FOLD into EV-2/`EVd-1`/`AR-6`; retain current secret redaction rule. |
| X0231–X0240 — event correlation | W1 `CX-2`, `EV-2`; S `AR-5`; W6 `EVd-1` | Vendors have separate session IDs; xencode needs its own run and worker IDs. | `CX-2` schema and the S-0 session table. | Stable `AR-5` worker IDs and `EVd-1` run identity. | Join evidence across tools, sessions and workers without merging vendor transcript stores. | M | H | M | REFINE into `AR-5` + `EVd-1`; define identifier links, not a new event database. |
| X0241–X0250 — latency analysis | W1 `CX-1`/`L-9`; S `AR-6` | Codex/Claude and other vendors may report their own timings/costs, with provider-specific completeness. | Existing rollup covers local xencode turns; no cross-agent usage run. | `AR-6` per-worker timing; `CX-1` aggregation; price source for cost. | Compare observed worker wall time and cost across vendors. | S | M | M | FOLD into `CX-1` and `AR-6`; require source and coverage on every figure. |
| X0251–X0260 — failure attribution | W1 `EV-2`, `EV-8`; S `AR-4`, `AR-6`; W6 `EVd-3` | A process exit, stream error, vendor-reported failure and verification failure are distinct signals. | EV-2 provider-error flag; replay coverage; `OR-16` envelope specification. | Normalized process and event states, then verifier result. | Separate worker claims, process outcomes and xencode verification evidence. | M | H | H | REFINE into `AR-4`/`OR-16`; do not collapse all failures into one status. |
| X0261–X0270 — trace sampling | None at present; local JSONL traces are bounded by display reads, not sampling | Vendors may sample or retain their own telemetry; xencode has no scale evidence yet. | No measured volume or storage pressure in the project. | Demonstrated storage/throughput limit and a retention policy. | Xencode could bound its own cross-worker traces, but no evidence supports sampling now. | M | L | M | RESEARCH MORE only if measured volume makes retention a problem. |
| X0271–X0280 — privacy-aware telemetry | W0 `DB-5`; W1 trace redaction in EV-2/QA-3; W7 `SE-5` | OpenTelemetry supports optional content capture; this can include prompts, tool args and results. | Existing `redact_secrets`; [OpenTelemetry GenAI overview](https://opentelemetry.io/blog/2026/genai-observability/); SE-5. | Data classification, redaction, opt-in and export controls. | Protect cross-worker data, enforce minimization, and make content capture explicit. | M | H | H | REFINE into SE-5 and any future exporter; default to metadata, never full content. |
| X0281–X0290 — export pipelines | No OTel exporter in tree; W14 has no existing telemetry transport item | Vendors can export telemetry; OpenTelemetry GenAI conventions are active and evolving. | OTel primary source says conventions are in active development and content capture is opt-in. | Stable internal event map, privacy policy, endpoint/auth config. | Export user-controlled, redacted telemetry if a concrete backend need appears. | L | M | H | RESEARCH MORE; no exporter or collector commitment without a user destination and data policy. |
| X0291–X0300 — observability queries | W1 `/trace`, `/cost`, `/profiler`, `xencode audit verify`; W6 ledger planned | Vendor CLIs provide logs/session views, but each is separate and vendor-owned. | Shipped CLI/TUI surfaces listed in W1 progress and S-0. | Reuse local trace/rollup; cross-worker query waits on `EVd-1`. | Query a joined engineering run across workers and xencode evidence. | M | M | M | FOLD into existing `/trace`/`/cost` and `EVd-1`; do not mirror vendor log browsers. |

## W04 — Agent Evaluation Lab

| candidate_id / theme | existing_overlap | vendor_overlap | evidence | dependency | xencode_ownership | complexity | value | risk | disposition |
|---|---|---|---|---|---|---|---|---|---|
| X0301–X0310 — task datasets | W1 `QA-5`, `EV-1` | Vendors provide their own eval/review surfaces unevenly; none replaces xencode's cross-worker task corpus. | Eight seeded Rust defect repositories, run through the real local agent loop. | Define tasks and grading before using external workers; `AR-1`. | Maintain neutral tasks that test cross-agent handoff and evidence. | M | H | M | REFINE as an EV-1 extension only when `AR-1` establishes a real adapter use case. |
| X0311–X0320 — trace graders | W1 `EV-1`, `EV-10`, `QA-3`; S `OR-16` | Vendor traces are proprietary formats; model self-report is not independent evidence. | EV-1 grades repository state; EV-10 judge-ranks near misses; OR-16 separates claims/evidence. | Normalized trace plus independent verifier. | Grade observed work, not narrative or inaccessible vendor internals. | M | H | H | FOLD into EV-1/EV-10/OR-16; avoid a second grader. |
| X0321–X0330 — behavioral tests | W1 `QA-5`, `EV-1`; W5 verification | Existing seeds cover eight defect classes and enforce a real diff plus passing tests. | QA-5 real seeded cases and EV-1 real agent loop. | Add only after a cross-agent behavior is specified. | Ensure a normalized interface preserves behavior across vendors. | M | H | M | FOLD into QA-5/EV-1; add cross-agent cases only for measured gaps. |
| X0331–X0340 — routing evaluations | W2 `AC-3`, `MI-7`; S `OR-1`, `OR-6`; W1 `EV-1` | Vendors expose their own model/agent selectors; no vendor chooses across competing external CLIs for xencode. | Existing rule-based task-shape routing and task eval; no external-worker routing trial yet. | `AR-1`, a defined task contract, then `OR-1` and `EV-1`. | Choose among workers only when capability/quality evidence supports it. | M | H | H | FOLD evaluation method into EV-1; defer implementation judgment until `OR-1` has measured tasks. |
| X0341–X0350 — handoff evaluations | S `AR-7`, `OR-15`, `OR-16`; W1 replay/eval | Vendors resume their own conversations; cross-vendor context transfer is not a vendor-provided guarantee. | S-2 vendor-local session boundary; no Xencode handoff experiment yet. | `AR-1`, then `AR-7` package and machine-checkable `OR-15`/`OR-16`. | Measure whether a neutral evidence package lets a second worker finish or verify work. | M | H | H | RESEARCH MORE; candidate is distinct, but needs an observed handoff and a scoring protocol before promotion. |
| X0351–X0360 — safety evaluations | W7 `SE-4`/`SE-5`/`CAP-1`; W1 `EV-1`; S `OR-3` | Gemini documents client-driven session approval-level changes; Claude's current help exposes a permission prompt target; neither has been exercised. These are different controls, and neither yet proves a complete xencode broker round trip. | S-3/S-13; 2026-09-27 CLI help and [Gemini ACP docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/acp-mode.md); no live permission request executed. | `AR-1`, permission-policy vocabulary and EV-1 verifier. | Test whether authority limits hold across workers and label the control mode honestly. | M | H | H | REFINE into `SE-4`/`OR-3` safety cases; update the old Claude-only claim only after live round trips are measured. |
| X0361–X0370 — regression suites | W1 `EV-8`, `QA-1`, `QA-2`; S `AR-1`, `AR-3` | Vendor versions have already moved since S-0; structured formats can drift. | Version deltas above and replay cassettes below HTTP boundary. | Captured real outputs and compatibility assertions. | Catch breakage in the xencode adapters and normalized schema. | M | H | M | FOLD into QA-1/AR-1; add only captured, provenance-bearing examples. |
| X0371–X0380 — benchmark runners | W1 `EV-1`, `QA-5`; W5 verification | Vendor benchmark suites are vendor-specific; none measures xencode's multi-worker value by itself. | `xencode eval run` and seed runner exist; no external-worker runs were done in this audit. | `AR-1`, bounded spend permission, reproducible tasks. | Compare complete engineering outcomes across workers under the same contract. | M | H | H | FOLD runner into EV-1; the cross-agent benchmark remains gated on AR-1 and spend approval. |
| X0381–X0390 — evaluation reports | W1 `EV-1`, `EV-10`, `/trace`, `/cost`; S `OR-16` | Vendors show their own run reports; cross-vendor attribution belongs to xencode only after event normalization. | Existing eval JSONL and EV-10; current vendor boundaries from S-0. | `AR-9` and consistent run identity. | Report outcomes with evidence, sample count, model/worker and permission posture. | S | M | M | FOLD into EV-1/EV-10/OR-16; no separate report format until a missing field is evidenced. |
| X0391–X0400 — quality gates | W5 `L-7`, `VF-*`; S `OR-17`; W6 `EVd-3` | Vendor exit statuses or “done” messages cannot independently certify project quality. | Existing exit-code gate plan and OR-16 evidence envelope; no cross-vendor task executed. | Verification engine W5, then evidence ledger and veto authority. | Let xencode block merge on project-owned checks and auditable evidence. | M | H | H | FOLD into L-7/EVd-3/OR-17; keep the worker itself unable to clear a veto. |

## Audit result

- **40 theme groups** cover all **400 IDs** in W01–W04; each group preserves its
  ten source IDs in `candidate_id`.
- Dispositions: **19 fold**, **15 refine**, **4 research more**, **0 promote**,
  **2 reject**. “Fold” means the existing roadmap owns the work; it does not
  imply every planned item is already implemented.
- No new implementation IDs are proposed. Research-more findings are
  impersonation defense (X0171–X0180), trace sampling (X0261–X0270), telemetry
  export (X0281–X0290), and handoff evaluation (X0341–X0350). Credential
  brokerage (X0131–X0140) and identity rotation for vendor accounts
  (X0151–X0160) are rejected because those credentials belong to the vendors.
- The existing Milestone S statement that vendor CLIs share “one event schema”
  is too strong: help output shows several structured formats, no evidence of
  equivalent event meanings, and one probed CLI without a documented structured
  output flag. Treat that claim as unverified until AR-1 captures and compares
  real streams. Gemini's documented session-mode control and Claude's current
  prompt-target option make S-13's Claude-only statement stale as a description
  of current CLI surfaces. They do not establish a complete xencode-controlled,
  per-request approval round trip.
- T-1 does not cover T-2's ten investigations in depth. It supplies the compact
  candidate crosswalk and identifies the measured questions T-2 should answer.

## Primary references

- Repository evidence: `NEXT_PLAN_TASKS.md` §§ Milestones M, S and T; W1 progress
  for EV-1/EV-2/EV-11/WF-1/CX-1; W7 trust plan; Rust modules named in those items.
- [MCP 2026-07-28 specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- [A2A v1.0 specification](https://a2a-protocol.org/v1.0.0/).
- [OpenTelemetry GenAI observability overview](https://opentelemetry.io/blog/2026/genai-observability/).
- [NIST agent identity concept paper](https://csrc.nist.gov/pubs/other/2026/02/05/accelerating-the-adoption-of-software-and-ai-agent/ipd) (initial public draft).
- [OpenAI Codex App Server article](https://openai.com/index/unlocking-the-codex-harness/).
- [Gemini CLI ACP mode documentation](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/acp-mode.md).
- [Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage).
- [Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage).
