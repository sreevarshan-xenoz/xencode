# Changelog

All notable changes to the Xencode project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — the file watcher now reports what it could not do
Two ways this feature could fail without saying anything. If the workspace could
not be watched at all — on a repository this size, usually because the operating
system's limit on watched paths is already used up — the watcher quietly gave up,
and "nothing changed" looked exactly like "watching is off". It now says so once
in chat: `⚠ file watching is off: <reason>`. Second, the watcher waits for a pause
in file activity before reporting a batch of changes, and a large pause never
came: a `git checkout` or a build that keeps touching files held the batch
indefinitely. The pause it will wait for is now capped at one second, and a batch
that has been growing for two seconds is reported even while activity continues.
Asking for a longer pause than the cap no longer delays a warning by that much,
because changes from one save arrive within milliseconds anyway.

### Changed — the retrieval scorecard measures something harder
`/ctx eval` grades the file finder against a set of questions whose answers are
known-good files in this repository. One of those questions pointed at
`cmd_output.rs`, a file that has never existed, so it could never be answered and
quietly pulled every score down; and all ten questions were answerable from a file
name alone, which is not a question worth asking a finder. The corpus is now
eighteen questions. The unreachable one was retargeted to the file that really
does trim command output, and the additions are phrased the way people actually
type — including negations and conditions: "secret files are flagged without their
contents being read", "refreshing a path that is not rust does nothing". Expect
lower numbers, and read them as the honest size of the gap: on a real index of
this workspace the deterministic pass scores 0.28 at rank 1 and 0.50 within the
top five, the hybrid pass 0.39 and 0.50. Every one of the eighteen answers still
comes first when it is put against three unrelated files, so the misses are about
telling files apart across a whole repository rather than about the questions
being wrong. A test now opens each gold answer on disk, so a question pointing at
a missing file fails the suite by name.

### Fixed — the security scan no longer flags ordinary code for using common words
`xencode analyze` reported High-severity path traversal (CWE-22) on lines that
merely contained the word `input`, and Medium-severity SSRF (CWE-918) on any line
containing `url` or `user`. A function declaration like `fn parse_input(raw:
&str) -> String {` was enough to produce a finding, so the scan's output could not
be believed. In both patterns the words were listed outside the group they belong
to, so the pattern language read them as separate, independent searches rather
than as arguments that must appear *inside* a file or request call. Both are
grouped correctly now. On a sample of five single-line files, six findings came back
before the change and two after — and the two that remain are the real ones, an
`open(user_path)` and a `fetch(url)`. Four tests pin both directions, each
re-checked against the old pattern to confirm it would have caught this.

### Fixed — a crash in the interface gives your terminal back and leaves a trace
When the TUI hit a bug it panicked with raw mode and the alternate screen still
switched on, so its own error message went onto a screen you could no longer
scroll or type into, your shell came back with keypresses misbehaving, and
nothing recorded that any of it had happened. A crash now puts the terminal back
first — normal input, no mouse capture, cursor visible — and writes the message,
the source location, and a backtrace if `RUST_BACKTRACE` was set to
`~/.xencode/last_panic.log`, readable only by you. The file is there for the
next release's health check to report; until then it is the one place a crash
can be described after the fact.

### Fixed — a log left half-written by a crash is still readable
`.xencode/cache/metrics.jsonl`, the file the TUI profiler reads, is appended one
record per line. A process killed mid-append — or a disk that filled — used to
leave a final line that was only part of a record, and a reader had no way to
tell that interrupted line from a record that was genuinely wrong. Reading a
log now drops the partial last line and reports that it did, while a corrupt
line anywhere else in the file is counted rather than passed over, because that
says something is broken in the writer. The profiler keeps the turns it had
already recorded.

### Fixed — your configuration file is no longer readable by everyone
`~/.xencode/config.json` holds provider API keys as plain text, and every
version so far wrote it with the operating system's default file permissions —
`644`, readable by any other user on the machine and by anything that can read
a backup or synced copy of your home directory. The manuals told you to
`chmod 600` it by hand; nothing stopped the next save from leaving whatever
mode the editor happened to create. Saving now goes through one helper that
writes to a fresh private file in the same directory, flushes it to disk, and
renames it into place, so a crash part-way through leaves the previous config
intact rather than a truncated one, and the file that appears is readable only
by you. A config that was already group- or world-readable is tightened the
next time a setting is saved. The same helper now backs every other piece of
state Xencode keeps on disk — the response cache, the project index and
transcript under `.xencode/`, conversation memory, Colab session state, and the
background-task registry that other `xencode tasks` processes read at any moment.

### Docs — what the orchestrator proposal was still missing (Milestone S revision)
The response to the recorded appendix added four items the original thirty-eight did not
contain, and they are the parts the idea needed to be an architecture rather than a
feature list. A common event protocol is now its own piece of work and the centre of the
runtime: six vendors can only be coordinated by something they all normalise into, and the
adapters are thin precisely because that target exists. A task contract fixes what "done"
means before a worker is launched — allowed and forbidden files, expected output, the
verification commands, the completion condition — because the everyday failure of this whole
feature is an agent that reports itself finished when nobody ever stated the condition. A
result envelope keeps a worker's claims in a separate field from its evidence, so a
reviewing agent receives a machine-readable engineering record instead of its predecessor's
prose. And a veto that the blocked party cannot lift, clearable only by a named reviewer, a
human, or a policy that says what it clears: that is the one kind of authority no vendor has
ever had over another, and therefore the honest answer to why anything belongs in the middle
at all. Parallelism stopped being a configured maximum and became a computation over whether
tasks are independent, whether their file sets collide, and what they cost.

Two rules arrived with them. "Do not scrape terminal output" is now a fallback order rather
than a preference: read a worker's structured stream, otherwise derive from repository state
ourselves, never scrape a terminal for text. And the rule the feature will be judged on —
live control over a worker's permissions is real for exactly one vendor today, while every
other vendor only accepts policy fixed before it starts, so the interface must display which
of the two it is looking at. Claiming the first when holding the second is how a safety
feature becomes a false assurance, and it is the same standard every other panel in this
project is already held to. What comes next is deliberately not a user interface: the
measurement, then the protocol derived from it, then the state model, then the task graph.
Plan items in the wave order: 284.

### Docs — the multi-agent orchestrator proposal, measured (Milestone S)
A third external proposal list arrived: thirty-eight items describing an
`/orchestrator` mode that would run Codex, Claude Code, Gemini CLI, OpenCode and Agy as
workers under one plan, one permission broker, one shared memory and one merge decision.
Unlike the earlier lists, this one could be checked against reality rather than argued
about, because all five of those programs are installed on the machine this project is
developed on — so the appendix records a table of what each one's own help text actually
advertises, read today. Three results reshaped the idea. The five have already converged
on the same headless contract (single-prompt mode, a machine-readable event stream, the
same three-rung approval ladder, resumable sessions), which makes the proposed adapter
layer a normaliser over command-line flags rather than five separate protocols, and drops
three of its twelve proposed methods because no vendor implements them. Roughly a third of
what was proposed is already shipped by the vendors being orchestrated, including their
own session daemons, subagent definitions, non-interactive review commands and health
checks — and Claude Code can import another agent's configuration wholesale. And the
permission broker, which the proposal treats as the centrepiece, has exactly one working
seam today: one vendor will route its approval requests to a model-context-protocol tool,
which is the same surface task M-5 already plans to expose, so that task moves earlier in
the order and every other vendor gets policy granted before launch — which is not control
and is now described as such. The proposal does reverse one standing rejection, on grounds
the plan accepts: the agent task graph was parked because a single local model saturates
this laptop, and workers calling a remote service do not. What replaces that ceiling is
written down instead of glossed over: one machine verifying everyone's output, a small
local model doing the planning, and real money spent per fan-out. Thirteen proposals were
already in the plan under another name, thirteen survive in a narrowed form, ten are new,
two are rejected, and twenty-two new task IDs join the wave order — the first of which is
a measurement: run each installed agent headless on a tiny read-only task and record what
its event stream actually carries. Nothing else in the appendix may be treated as a
commitment until that matrix exists. Nothing here is built; the wave count in Milestone R
grows from fifteen to eighteen accordingly.

### Docs — the plan finally has an order (Milestone R)
Four research appendices had declined to rank their candidates, which was honest
about value but left no way to start anything. The owner supplied an ordering
instead of a ranking: fifteen dependency waves over all 258 recorded plan items,
from "fix what makes today's output untrustworthy" through observability, the
model layer, code intelligence, verification, trust and durable knowledge out to
the product surface. Each item is placed exactly once and keeps the identifier the
appendix that produced it gave it, so provenance survives ordering; the placement
was checked by script rather than by eye. Seven of the proposed placements moved,
because the plan already fixes some dependencies in ink — most importantly, the
project-notes writer cannot be a first-day item, since it must not exist before
the knowledge source-class model that stops it turning a bad summary into durable
truth. One item is parked as genuinely contested rather than resolved: speculative
decoding, which an earlier milestone declined for this laptop's hardware and a
later one re-proposed for a rented GPU. What the waves deliberately do not decide
is which item in a wave is worth doing first.

### Docs — the commit rule now names what a commit completes
The standing rule said "commit each change before the next". With an ordered plan
that was too loose to audit, so `AGENTS.md` now defines the unit as a single plan
item rather than a wave, requires every commit message to name the item or items it
completes, requires both identifiers when two items turn out to be one change, and
makes wave completion traceable — a wave is done only when each non-deferred item
in it can be matched to a commit naming it. It also states plainly that commit
messages are user-facing product history and must be written in plain, fully named
English, with a list of the internal shorthand that is not acceptable in them.
Nothing about the code changed.

### Docs — Milestone Q: the second hundred, dispositioned
A second external proposal list (100 items around project DNA, time, system,
trust and experimentation) was checked item by item against the Rust tree rather
than adopted: 25 were already planned under existing IDs, 35 narrowed, 10 new,
30 rejected, recorded as 25 do-not-build register rows, and taking the research
pool from 176 to 208 options — **unranked**, as with N, O and P. Four of the
list's premises did not survive contact with this machine: it assumes no GPU (the
box has an MX250, 2 GiB), the plaintext-key `config.json` at mode 644 is
described as a missing capability when the OS keyring is present and simply
unused, the eval fixture `gold.json` names a `cmd_output.rs` that is not in the
tree, and two `xencode-analysis-rs` security regexes group their alternation
wrong, so any line containing the bare word `input` — or `url` — reports High
severity. Nothing was built; recording the defects is not fixing them.

### Docs — two promises the code never made
The research passes behind Milestones N–P checked manual claims against the
implementation, and two did not hold. `README.md` advertised "error
classification and targeted fix suggestions": there is no classifier — tool and
shell failures reach the model as `error:`/`exit <code>` results and the agent's
instructions tell it to name the failure — so the bullet now says that, and says
the classifier is planned rather than present. The Roadmap section claimed there
was "no open milestone" and "no known gap between promise and code"; five
planning tracks (L, M, N, O, P) are recorded, and the same passes listed live
gaps as defects. The stale crate count (14 → 15) is corrected with it.

### Docs — the product description catches up with the product
Milestone K made the model endpoint something you point at, so "offline-first"
in the tagline, the highlights list and the privacy row no longer described the
build. Replaced with "local-first, bring your own model" in `README.md` and
`docs/USER_MANUAL.md`, and the GitHub repo description with it. The absolute
"works with no network" claim is deliberately not made — that needs the offline
conformance test (plan item LF-8) before it can be asserted.

### Milestone K — remote providers + Google Colab — complete ✅
A GPU you do not own as an inference backend: `remote:` routes any
OpenAI-compatible endpoint (dedicated to the Colab SSH-forward case in the
docs), Settings gains real provider-key editing with masked `Secret` rows,
and the Colab bridge gets its preflight gate. The connectivity decision that
shapes the milestone: single supported transport is the official
`google-colab-cli` `colab ssh --proxy-mode` WebSocket SSH bridge — no public
tunnel, which Colab's free tier forbids (account suspension). Version trap
caught in code: `google-colab-cli` 0.6.0 shipped without the `ssh`
subcommand (upstream issue #102); preflight requires >= 0.7.0.

### Added
- **Colab bridge proven on a live VM, and fixed by what the run showed**
  (K-2c/K-3 follow-up). A free-tier T4 was brought up, served a Q4_K_M GGUF
  through llama.cpp, and answered real `xencode query -m 'remote:…'` calls over
  the forward; Provider Health went green, `up --reconnect` restored a killed
  forward, and `down` left no session and no orphan process. Six defects came
  out of that run: the bridge now logs in as **root** (Colab injects the key
  for root only, so the old default could never authenticate), the VM-side port
  moved to **18080** (Colab's own proxy permanently holds `8080`), the
  bootstrap reports `READY` only once `/v1/models` serves (a 0.5B model took
  ~39 s to load, so a spawn-time `READY` raced the probe), `--quant` /
  `config colab_quant` choose the GGUF quant, bridge-slot errors
  (`Already-active SSH session`, `banner exchange` timeouts) are retried rather
  than failing the bring-up, `--reconnect` tries the forward before re-running
  the bootstrap (9 s vs a full re-download), and the detached forward no longer
  inherits stderr — a piped `xencode colab up | grep` used to hang until the
  tunnel died.
- **Colab survivability: 12-hour-reap detection + one-key reconnect** (K-3).
  `xencode colab status` now compares the recorded `started_at` age against
  the endpoint: older than 12 hours (Colab's per-VM runtime limit) with a
  dead forward is reported as a reaped VM, with the exact recovery command.
  `xencode colab up --reconnect` rebuilds the bridge from `colab.json`
  alone: a live endpoint short-circuits to its URL (zero `colab`/`ssh`
  calls); otherwise the session is re-created if Colab reaped it, the
  bootstrap re-runs, and the forward re-spawns until `/v1/models` answers —
  no re-typing of the original flags. Missing `colab.json` fails fast with
  a fix suggestion. The TUI Provider Health panel gains a Remote/Colab
  forward row: seeded like the other providers (unconfigured remote URLs
  read "Remote URL not configured (Settings → Remote URL)"), the health
  check probes `{remote_base_url}/models` with a 5s cap, and Connection
  Details lists the configured Remote URI.
- **Colab VM lifecycle** (K-2c). `xencode colab up` brings a Colab VM up
  end-to-end: creates the session (`colab new --gpu <gpu> -s <name>`) when
  absent, pushes an ssh bootstrap that installs and starts the chosen runtime
  bound to `127.0.0.1` only (llama.cpp — pinned prebuilt llama.cpp release +
  one HF GGUF on 18080 — or ollama — `ollama serve` on 11434, its tags flow into
  the model picker), holds the `-N -l root -L` forward, waits for `/v1/models`,
  writes `~/.xencode/colab.json`, and points
  `llama_cpp_url`/`ollama_url`/`remote_base_url` at the forward.
  `xencode colab status` reports forward
  pid / `colab sessions` listing / endpoint probe and never fails hard;
  `xencode colab down` is idempotent (kill forward, `colab stop`, clear
  state). Flag > `config colab_*` fallbacks; session names validated before
  any shell use.
- **Colab preflight gate** (K-2a). `xencode colab preflight` verifies in one
  pass that the bridge is usable before any VM is brought up: the `colab` CLI
  on PATH, version >= 0.7.0 (version string *and* a functional `colab ssh
  --help` probe catch the 0.6.0 trap), backend auth via `colab sessions`,
  ssh/ssh-keygen on PATH, and an ed25519 key pair under the xencode config dir
  (`xencode colab preflight --generate-key` creates it). Each failing check
  prints a runnable fix; the command exits non-zero when any check fails.
  Lives in the new `xencode-colab-rs` crate.
- **`xencode config set remote_url` / `remote_key`** (Milestone K ¶). Any
  OpenAI-compatible endpoint can now be pointed at from config — the Remote /
  Colab provider kind routes through `OpenAICompatibleProvider` in every
  `ProviderManager` path (`remote_base_url` + `api_keys.remote_api_key`).
- **Settings edits provider endpoints and keys** (K-1b). The Providers section
  edits real values instead of reporting key presence: Remote URL as a text
  row, Remote/Gemini/Qwen/OpenRouter keys as masked `Secret` rows with a
  last-four tail on display. Commit saves through `App::save_config()`;
  empties clear the key; `Esc` discards the editing buffer so a plaintext key
  never lingers.

### Milestone J — complete ✅
Seven TUI panels rendered hardcoded phrase lists (voice, terminal assistant,
security auditor, profiler, custom models, learning mode, multi-language) and
the plugin registry loaded no runtime. All eight items landed — the
security auditor and the profiler are real, the terminal assistant delegates to
a model through the agent's approval gate, the multi-language panel tabulates
a real `scan_tree` walk and translates through a real model call, the custom
models panel edits `model_profiles` in `config.json`, the learning mode
panel teaches files the project index actually found, the voice panel
records from the microphone and keeps a WAV, transcribing only when a whisper
CLI is installed, and a manifest now loads for real (J-08). No TUI panel ships
scripted content and no plugin manifest claims a capability this build lacks.
`NEXT_PLAN_TASKS.md` §
Milestone J tracks the work item by item, with a done-when rule per panel.

### Added
- **Plugin runtime** (J-08). `xencode_plugin_rs::PluginRuntime::load(dir,
  xencode_version)` discovers `plugin.json` / `manifest.json`, skips a manifest
  whose `xencode_version` does not accept this build (reported, not silently
  dropped), registers each remaining one with the `Host`, and flattens the
  session into the two effects a plugin may have: a `prompt_prefix` and
  `before`/`after` hooks. Plugin hooks only land where `agent_hooks` in
  config.json is silent, so the config always wins and the approval gate is
  untouched. There is no dynamic linking and no plugin code — `entry_point` left
  with the Python runtime it named.
- `xencode plugin list` / `install` print what actually took hold (`guardrails
  v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)`, or
  `NOT LOADED: needs xencode 0.1.0 (declared 9.9.9)`), and the TUI gained
  `/plugin [reload]`, which reports the same verdicts plus the hook and prompt
  totals in effect for the session.
- New `default_plugin_dir()`: `$XCODE_PLUGIN_DIR`, else
  `<data dir>/xencode/plugins` — the directory the CLI installs into is the one
  the TUI loads from at startup.

### Removed
- **Unused deployment surface and stale design docs.** Deleted `k8s/`
  (`deployment.yaml` pinned the Python API's port 8000, `postgres.yaml` and
  `secrets.template.yaml` fed `DATABASE_URL`/`REDIS_URL`/`JWT_SECRET_KEY` to
  nothing) and `monitoring/prometheus.yml` (the Rust server exposes no metrics
  route). `.github/workflows/ci-cd.yml` lost its `deploy-staging` and
  `deploy-production` jobs and its `k8s/**` path trigger — it is now gate →
  image build → push → Trivy; `ci.yml` remains the plain test gate. Also ten
  Python-era documents under `docs/` (analytics, security-scanning,
  performance dashboard, provider-health/phase-3/validation/terminal-test
  summaries, `features/benchmark-wizard.md`, `phase5/fallback-engine.md`), all
  of them describing modules and `localhost:8000` routes that are not in this
  tree and linked from no current doc.
- **Dead Rust, deleted with no replacement.** `xencode-context-rs`
  `cmd_output` (a command-output blob index that nothing ever wrote to or read
  from), the never-implemented `Embedder` extension trait with its `cosine`
  helper and `Bm25::max_score_per_doc`, `expand_dependencies`, and the unused
  `CancelFlag` alias. `xencode-analysis-rs` lost `embeddings`
  (`EmbeddingClient`), `vector_store` (`VectorStore`, `cosine_similarity`) and
  `indexer` (`ChunkIndexer`, `DocumentChunk`) — the retrieval path in
  `xencode-context-rs` never called them. Dropped the `serde_json` and `uuid`
  dependencies those modules were the only users of; suite is 684 tests.
- **Python-era tooling configs.** `.pre-commit-config.yaml` (bandit, ruff,
  mypy, safety, detect-secrets against a baseline file that does not exist) and
  `.bandit` (a findings baseline for code that is no longer in the tree). The
  gates are `cargo fmt --check`, `cargo clippy --workspace --all-targets` and
  `cargo test --workspace`, which CI already runs. Also untracked
  `.xencode/technical_debt.db`, a 274 KiB SQLite artifact of the Python-era
  `technical_debt_manager.py` (`debt_scans` / `debt_items` tables) that no Rust
  code opens — `.gitignore` already covers `.xencode/`.
- **The Python-era documentation archive, deleted rather than kept as history.**
  `DOCUMENTATION.md` (737 lines: a "dual-stack architecture — Python feature
  system + Rust core runtime", a credential vault, ensemble reasoning, 12
  crates/65 tests), `PRD.md`, `project details.md`, `docs/FEATURES.md` (a
  "wild ideas" backlog whose migration status line still said 12 crates/65
  tests), `docs/ARCHITECTURE_DIAGRAMS.md` (mermaid diagrams of an API Gateway, a
  Connection Pool module and a Distributed Cache — the Python package shape, not
  the 14-crate workspace) and `BROWSER_LOGIN_PLAN.md`. Also
  `docs/superpowers/` (4,282 lines of executed migration plans, their specs and
  agent-worker instructions, linked from nothing) and three unused screenshots
  (`images/4-6.jpg`; the README shows 1-3). The README archive table that pointed
  at them is gone, `docs/INSTALL_MANUAL.md` and the README's architecture pointer
  now point at real files, and `docs/ROADMAP.md` was rewritten from the tree:
  its `xencode --git-commit` / `--git-review` / `--git-branch suggest` flags and
  `/analyze` / `/smart` chat commands never existed in the Rust CLI (the real
  entry points are `xencode review` and the TUI's `Ctrl+R` / `Ctrl+Y` /
  `Ctrl+S`), and `docs/api_documentation.md` lost its second half — the
  `xencode.core.*` module reference and Python usage examples (307 → 48 lines,
  the server's auth matrix kept). 9,371 lines of removed docs in total.

### Fixed
- **`xencode plugin remove <name>` accepted a path.** The name was interpolated
  straight into the plugin directory, so `../../etc` resolved outside it. Removal
  now goes through the same `PluginRegistry::plugin_path` guard the installer
  uses and rejects an invalid name (`fix(plugin)`).
- **Provider fallback chain (I4-01)** now honours its own eligibility rule:
  `retry::is_fallback_eligible` shipped with the feature but was never called,
  so a response xencode could not decode walked every configured alternate
  even though each reproduces the failure. The chain advances only on a clean,
  eligible error (`fix(agent)`).
- **TUI tests are hermetic.** `App::new()` read *and* wrote the developer's
  real `~/.xencode/conversation_memory.json` and restored its last messages,
  so a full-workspace run could fail a chat test with another test's transcript
  (`left: "/init abort", right: "/mcp status"`). Added `App::for_tests()`
  (non-persistent memory), made `ConversationMemory::memory_dir()` respect
  `XCODE_CONFIG_DIR`, and stopped persisting slash commands to conversation
  memory — a local TUI verb used to be replayed to the model forever. Tests no
  longer write the developer's real `~/.xencode/config.json` either:
  `App::for_tests()` turns config persistence off (`fix(tui)`).

### Documentation
- **README re-verified against the code**, not just re-worded. Corrected: the
  chat-input command list said "eight" while `SLASH_COMMANDS` has nine, and the
  profiler's metrics file was given as `.xencode/metrics.jsonl` instead of
  `.xencode/cache/metrics.jsonl` (`context-rs::metrics_path`); the prerequisites
  table called Ollama "required" and pinned an "Rust 1.80+" MSRV that nothing in
  the tree declares (CI builds `stable`). Added: the 16 real CLI subcommands,
  `xencode query`'s llama.cpp sampling flags, what `install.sh` / `install.ps1`
  actually do and where each puts the binary, the `release.yml` tag →
  smoke-test → GitHub Release path, the seven Milestone J panels described
  per-panel, and a "nothing scripted" highlight. The Roadmap section stopped
  listing shipped work (retry budgets, provider health UX, multimodal, team
  workflows) as future and now names what is genuinely unbuilt — TTS output,
  arena mode, coordinating several agent runs, AI commit messages, session
  replay — with the Anthropic key and CRDT wiring called out as parked decisions.
- Milestone I close-out honesty sweep (I4-02): every manual was re-checked
  against the tree. Removed fiction: the credential vault and `xencode vault
  init|migrate|status` (no such commands), ensemble/multi-model "reasoning"
  (the fallback chain is sequential), "language-aware AST analysis"
  (per-language heuristics), in-chat `/help /models /model /project /status
  /clear /exit` (eight real slash commands), a first-run setup wizard (there is
  none), Python-era setup and linting in `CONTRIBUTING.md` (`venv`,
  `requirements.txt`, `pytest`, `ruff`, `mypy`, `bandit`, a `dev` branch), and
  `.xencode.example.json`, which described a Python config shape and is now the
  real flat `config.json`, validated through the loader.
- Stated plainly what is not real: seven TUI panels (voice interface, terminal
  assistant, security auditor, performance profiler, custom models, learning
  mode, multi-language) render scripted content; the plugin commands manage
  manifests with no runtime that loads `XencodePlugin`; `GET /api/models`
  appends hardcoded remote entries; `.xencode.example.json`, which described a
  Python config shape and is now the real flat `config.json`. Each of those
  gaps was closed by Milestone J (J-01…J-08) in this same unreleased cycle.
- Recorded a routing gap: `xencode-providers-rs` has an Anthropic client but
  `ApiKeys` has no `anthropic_api_key` and both entry points pass `None`, so an
  `anthropic:…` model cannot authenticate — Claude ids work through
  OpenRouter.
- Counts and flags corrected: 14 crates, 692 tests; `xencode query --session`
  (not `--session-id`); `xencode llamacpp list|set-path`, `cache clear`,
  `memory show <id>`; `models health` requires a model name; README version
  badge back to the real `0.1.0`.

### Added
- Voice Interface is real (Milestone J, J-07). `Enter` used to play a scripted
  session: four hardcoded phrases with invented results (`run tests` →
  "✅ 142 tests passed, 0 failed"), an audio level cycling through
  `0.3, 0.6, 0.8, 0.9, 0.7, 0.4` regardless of any microphone, and a "speaking"
  state for a product with no text-to-speech. Now `Enter` spawns the first
  recorder found on `PATH` (`arecord`, then `pw-record`, then `parec`) with a
  fixed argv that streams raw 16-bit mono 16 kHz — no shell string, so this
  never touches the agent's `run_command` approval path — and every 100 ms
  chunk of what it actually sent yields one RMS reading, which drives the level
  bar, the session peak and the clip length. `Enter` again, or `Esc` while
  recording, ends the capture early and keeps the clip; `m`/`Space` mutes by
  discarding audio rather than saving a silent file. The PCM is written to
  `.xencode/voice/clip-<unix>.wav` through a hand-built RIFF header. Speech text
  comes only from a whisper CLI's stdout (`whisper`, `whisper-cpp`,
  `whisper-cli`); with none installed the panel names the clip it kept and says
  there is no speech engine, and an engine that fails, prints nothing or dies
  contributes its own error or nothing at all. Deleted with this: the canned
  phrase list, the fake level cycle, and the dead `voice_commands`,
  `voice_confidence` and `voice_language` state. Verified against the real
  microphone path — `arecord` on this machine produced a valid 1.85 s
  16-bit mono WAV and the no-engine note — and covered without a microphone by
  nine `voice` module tests that substitute `cat` on a prepared PCM file, six
  app tests for the level/clip/error/mute reductions, and three panel renders.
- Learning Mode panel is real (Milestone J, J-06). It opened with one invented
  lesson — "Rust Ownership Basics", a `calculate_length` snippet, an "exercise"
  with nothing to submit it to — and a quiz it marked correct whenever option 1
  was picked, because `learn_quiz_correct` compared the selection to `0`. Now
  `Enter` reads `.xencode/index/symbols.json` and queues the files that declare
  something (most declarations first, ties by path, five at a time), so the
  lesson list belongs to this workspace: a file with no declarations is not a
  lesson, and a workspace with no index gets "No project index — run /init
  first" and no request. Each lesson shows that file's own text — capped at a
  line boundary, with a note saying how much of it was sent — next to the
  declarations the index recorded, then makes one model call asking for
  `{explain, question, options, answer, why}` about that exact text. The
  model's sentences are drawn under "The model says:"; the answer key is the
  model's, the panel names which option was the key and prints its reason, and a
  reply with no usable key in it is shown as the reply instead of becoming a
  canned question. `p`/`n` walk the queue, `r` re-asks. Dropped `learn_exercise`
  and the `learn_progress_pct` score (progress now means position in the queue);
  fixed the header box, which fitted 1 of its 2 lines so the count never
  showed.
- Custom Models panel is real (Milestone J, J-05). Pressing `Enter` used to seed
  four invented profiles (`Code Assistant`, `Creative Writer`, `Bug Hunter`,
  `Code Reviewer`) with sliders and a "Profile saved!" line that wrote nothing.
  The panel now lists `model_profiles` from `config.json` — a new
  `ModelProfile { name, model, temperature, max_tokens }` in `xencode-config-rs`,
  `#[serde(default)]` so an older config still loads — and an empty list renders
  "None yet — press n" instead of samples. `n` adds a profile from the current
  session's settings, `-`/`+` moves temperature over 0.0–2.0, `←`/`→` steps the
  token budget along 64…8192, and an unset knob starts from the value the session
  would send anyway. `Enter` applies to the next turn in memory only (model id,
  the llama.cpp knobs, the selector's position, plus a `switch` when the id
  points at the llama.cpp server); `s` is the only key that writes `config.json`,
  and it reports what happened — `wrote N profile(s)`, `config.json unchanged:
  {error}`, or that persistence is off in this session. `t` sends one request
  with exactly that profile's settings through the shared `SingleShot` path, so
  the status line is the provider's own reply or its own error. `top_p` was
  **deliberately left out**: `merge_llamacpp_options` is the only place sampling
  knobs reach a request, and it carries temperature and max tokens to llama.cpp —
  nothing in this workspace sends top_p, so the panel states that llama.cpp is
  where these numbers land rather than implying Ollama and cloud endpoints obey
  them. Dead state removed with the seeds: `models_editing`, `models_saving`,
  `models_test_output`, `models_profiles` (the 5-tuple list) and
  `start_custom_models`.
- Multi-language panel is real (Milestone J, J-04). Its "language detection" was
  a fixed table (`python 34.2%`, `javascript 27.1%`, …) for files that are not
  in this workspace, and its "Supported Languages" list was hand-written. `Enter`
  or `d` now runs `scanner::scan_tree` on the workspace root in
  `spawn_blocking` and streams the languages that are *actually present* — file
  count, line count from the scanner's own `count_loc` (blanks and
  comment-leading lines excluded) and share of those lines — sorted by lines,
  then name. The notes under the table carry the walk's fine print: total
  files/lines, what ignore rules skipped, and the fact that secret and binary
  files are listed but never read, so they add files and zero lines rather than
  a made-up size. The language legend is
  `scanner::Language::ALL` (a new const, so the panel reads the enum instead of
  copying it), with `▸` marking what the walk found; two new scanner tests pin
  that `ALL` and `language_for_extension` agree in both directions and that
  every `as_str()` is unique and lowercase. Translation makes one real model
  call through the same single-shot path as the terminal assistant: `Tab` selects
  From / To / Text, typing edits that field, `Enter` asks the configured model
  and prints its reply — or prints the provider's error, drawn as an error.
  Empty text is refused without spending a request. Dead state went with the
  scripted table: `lang_active`, `lang_supported`, `term_asst_active`,
  `term_asst_cursor`, `open_terminal_assistant` and the sender-less
  `[TERM]output:` protocol arm.
- Terminal assistant panel is real (Milestone J, J-03). It used to open with
  five hardcoded "suggested commands" — including `docker system prune -af` and
  `rm -rf node_modules && npm install` — each with a risk badge, and nothing
  behind any of it. Now the panel opens as a question field: `Enter` makes
  **one** call to the configured model, whose prompt carries the workspace path,
  its top-level entries from the context index and the current git branch, and
  asks for a JSON array of `{command, risk, why}` capped at 8. Fences, prose and
  a bare object are all tolerated; a reply with no commands in it is shown as
  the reply rather than converted into invented suggestions. Risk labels only
  escalate — `DESTRUCTIVE_PATTERNS` overrides a command the model called safe,
  and a label the model raised is never lowered. `f` filters by risk, `j`/`k`
  select, `y`/`Enter` runs the selection **through `execute_tool_call_approved`
  with the session's `ApprovalCtx`** — the same policy, modal, hooks and
  checkpoint group as a model-issued `run_command`, with `approval_ctx()` now
  the single place that context is built. Outcomes, denials included, land in
  the panel's history; provider errors are printed, flattened to one line. A
  `Ctrl+T`-style shell path was deliberately not added: the panel has no route
  to a binary that the gate is not on.
- Performance profiler panel is real (Milestone J, J-02): the table used to list
  six invented functions (`process_data`, `query_database`, …) with random CPU,
  memory and latency numbers. `Enter` now measures: process CPU from two
  `/proc/self/stat` reads 250 ms apart, resident memory from `/proc/self/statm`
  against `/proc/meminfo`, and the numbers the session already has — average
  turn latency, llama.cpp tokens/s and token counts, per-provider health
  latency (or the error that stopped it), and the last 6 rows of
  `.xencode/metrics.jsonl` with KV reuse, prompt tokens, tok/s and retrieved
  files. A gauge with no measurement renders `n/a` rather than `0`, and the
  panel says when there has been no turn, no health check or no metrics file.
  `fastrand` left the dependency list with the random numbers. Fixed the bug
  that hid this: the Gauges box laid out 3 of its 5 lines, so the Memory and
  Latency gauges had never been on screen.
- Security auditor panel is real (Milestone J, J-01): `Enter` in the panel used
  to print seven hardcoded findings about a `config.py` that is not in this
  tree. It now walks the workspace with the same file list the context engine
  uses (`scanner::scan_tree`, off the UI thread) and runs
  `VulnerabilityScanner::scan_file` — the scanner behind `xencode analyze` — on
  every readable file, streaming each hit as
  `[SECURITY]finding:SEVERITY|type|file:line|message — recommendation`. Secret
  files the walk flags come in as Medium findings, the progress bar is files
  scanned over files found, and the log lines carry the real totals plus every
  file it could not read. If the walk itself fails the panel says
  `scan failed: <reason>` and goes idle instead of showing an empty scan as a
  clean one. Findings are capped at 200 on screen; the reported counts stay the
  true totals. `CodeAnalyzer` is *not* included — it reports style and
  maintainability issues, not security ones.
- `/spawn` subagent in a git worktree (Milestone I, I3-03): `/spawn
  <task> [#branch]` runs the delegated agent loop (`/bytebot`'s engine)
  inside a fresh sibling worktree next to the project — `<dir>-spawn-<id>`
  on the generated branch `xencode/spawn-<id>`, or `<dir>-spawn-<id>-<branch>`
  when the task ends with a `#branch`. The subagent gets its own tool root
  and a fresh checkpoint store (so `/rewind` in the main chat never touches
  its files) while your own chat keeps working. Live tool calls stream in
  the transcript; on completion the agent's final answer is posted as
  `(spawn #<id> · <task>)` and a `⏺ spawn #<id> done/failed — branch … at …,
  N/M call(s) completed` line names the worktree. `/spawn status` lists every
  registered run with its worktree location
- Agent pre/post tool-use hooks (Milestone I, I3-02): the `agent_hooks` config
  key declares shell commands that run around *approved* agent tool calls — a
  `before` map and an `after` map, each keyed by exact tool name or `*` for
  every tool. Commands run via `sh -c` in the workspace root with the same
  budget and output cap as `run_command` (stderr merged, tail kept). A
  `before` hook that exits non-zero **vetoes the call**: nothing runs, no
  rewind point is taken, and the model sees `error: pre-hook vetoed this
  call:` plus the hook's output. A passing `before` hook has its output
  prepended to the result; the `after` hook always runs and its output is
  appended. Hooks are config-edited in the JSON like `mcp_servers`
- MCP tool servers (Milestone I, I3-01): a new `xencode-mcp-rs` crate speaks
  the Model Context Protocol over stdio. Servers are declared in config under
  `mcp_servers` (`command`, `args`, `env` — credentials never in `args`) and
  are started only when you ask with `/mcp`; a broken or missing server reports
  its own failure in a `[MCP]✗ …` line instead of stalling the TUI. Each live
  tool is offered to the model as `mcp__<server>__<tool>` and routed through
  the same approval gate as every other agent tool (`External` class — always
  `y`/`n`, never waved through by autonomy), and `/mcp stop` withdraws the
  tools of everything it stops. `/mcp status` lists what is running; the
  per-handshake budget is `mcp_timeout` config (default 30 s, `config set
  mcp_timeout`, `1`–`300`)
- ByteBot is the real agent (Milestone I, I2-04): `/bytebot <task>` and the
  panel's Enter used to play a recording — six fixed steps, `sleep`s, and
  invented output like "🧪 Running test suite (142 tests)" and "✅ All tests
  pass". Both now run the *same* tool loop as a chat turn, with the task as its
  only user turn (project guidelines, git state and retrieved context, no chat
  history), so the panel's **Execution Steps** are the calls the model actually
  made: `⏳` while one runs, then `✅ done`, `✗ denied`, `✗ refused` or `❌
  failed`. The progress bar is finished calls over calls made — it can move
  backwards, which is honest. The approval gate is exactly your
  `agent_approval` setting (`ask` prompts every edit and every command), the
  writes are checkpointed so one `/rewind` undoes the whole run, a plan the run
  posts shows in the same strip, and a provider failure is printed as the error
  it was. `is_generating` no longer claims a delegated run, `/rewind` refuses
  while ByteBot is working, and the panel keeps the task visible while it runs
  instead of a stock "Executing…" line
- The agent's plan is now on screen (Milestone I, I2-03): `update_plan(items)`
  lets the model post its todo list — one `{text, status}` object per step, at
  most 12 — and it renders above the chat transcript as a `☰ Plan 2/5` strip
  with `✓` done (struck through), `▶` in progress and `·` pending. The compact
  strip shows the first 6 steps and says how many are hidden; `/plan` pins the
  whole list, `/plan clear` drops it, and neither opens a chat turn. Posting a
  plan is a read-only call, so it never costs an approval even in `ask` mode.
  Parsing is deliberately tolerant of what local models actually write (bare
  strings, `- [x]` markdown, invented keys like `title`/`state`, a JSON string
  instead of an array); a rejected update is an error the model can act on and
  leaves the visible plan untouched. On a pane too short for both the strip and
  six lines of transcript the strip is dropped rather than squeezing the chat
- `run_command` for the agent (Milestone I, I2-02): the missing `t` in
  edit→test→fix. The model runs one foreground `sh -c` line in the workspace
  root and gets back `$ <command>`, `exit <code>`, then the combined
  stdout+stderr — the last 8 KiB of it, because a failing build ends with the
  reason. It is a `shell command` class call, so the approval prompt shows the
  literal command line (there is no diff to show) and `edit-allow` still asks.
  `agent_command_timeout` (default 30 s; new `Command Timeout` Settings row in
  5-second steps, `xencode config set` accepts 1–600) kills anything that
  overruns and tells the model nothing was captured, pointing it at
  `background_start` for slow work
- Agent checkpoints and `/rewind` (Milestone I, I2-01): every write or edit the
  user approves is snapshotted first — the exact prior bytes, or "this file did
  not exist" — grouped by chat turn in memory. `/rewind` undoes the last turn
  that changed files, `/rewind 3` the last three; created files are deleted,
  changed files come back byte-for-byte, turns that wrote nothing are skipped
  rather than counted. Nothing touches git, nothing is persisted, and a file
  larger than 4 MiB says out loud that `/rewind` cannot undo it. It refuses to
  run mid-generation, and if the rewound file is open with unsaved edits the
  editor warns instead of discarding them
- The agent tool loop is now file-capable and gated (Milestone I, I1-04): the
  chat turn offers `file_tools()` next to the background and advice tools,
  and every requested call goes through the permission policy first —
  `Allow` runs it, `Deny` answers `error: refused by the permission policy`
  without ever prompting, and `Ask` raises the approval overlay and waits. A
  yes executes the call, an "always allow" answer also grants that tool class
  for the rest of the session (shared state, so the next message inherits
  it), and a no — or a prompt whose UI disappeared mid-wait — returns
  `DENIED_RESULT`, worded so the model explains instead of retrying the same
  call. The model is taught the vocabulary, the workspace boundary and the
  no-retry rule in a short tools block appended to the system turn. New
  `agent_max_rounds` config key (default 16, clamped 1..=64, settable with
  `xencode config set`) replaces the old hard-coded 8-round valve
- File tools for the agent (Milestone I, I1-02): `read_file` (paged, numbered
  lines), `list_dir`, `search_files` (regex walk that skips `target/`,
  `node_modules/` and dot-dirs, 100-hit cap), `write_file` and `edit_file`
  (exact old→new replace, `all` for multi-match) are defined in
  `xencode-providers-rs::file_tools` and executed in
  `xencode-tui-rs::agent_tools`, each returning a unified diff built with
  `similar`. Every executor refuses paths outside the workspace on its own,
  and `background_start` now resolves a relative `cwd` against the workspace
  root instead of the process directory
- Agent approval prompt (Milestone I, I1-03): when the policy says `Ask`, the
  TUI shows a topmost modal with the call, its class and the exact bytes at
  stake — colored unified diff for writes/edits, the literal command line for
  shell calls. `y` allows, `a` allows and remembers the class for this
  session, `n`/`Esc` deny, `k`/`j` scroll; every other key (quit chords
  included) is swallowed and Enter is deliberately *not* an answer. Stacked
  calls queue FIFO, denials come back to the model as a message telling it
  not to retry unchanged, and each answer is logged in the chat as
  `⚙ write_file src/lib.rs · approved`. The keys also appear in the `?` help
  overlay and the status bar
- Agent permission policy core (Milestone I, I1-01): every tool call is now
  classified against an `agent_approval` mode — `ask` (default),
  `edit-allow`, `all-allow` — via a single `classify` in
  `xencode-tui-rs::agent_tools`; read-only tools always run, file edits and
  shell calls prompt or auto-allow per mode, and paths outside the workspace,
  inside `.git/` or the config dir are hard-denied in every mode. New
  `Agent Approval` Settings row (Cycle) and `xencode config set
  agent_approval`; session "always allow" grants live in `App.agent_grants`
  until quit. Approval prompts and file tools land in I1-02…I1-04
- Selectable TUI body layouts and display preferences (Milestone H): three
  layout presets — `classic` (20/50/30 as before), `chat-first` (explorer
  hidden, editor 25 %, chat 75 %), `zen` (one pane fills the body, following
  focus) — cycled live with `Ctrl+U` or picked on the Settings panel, plus
  `rounded_borders`, `show_scrollbars` (chat & explorer) and
  `show_line_numbers` (editor gutter + current-line highlight). All four
  persist to `~/.xencode/config.json` and are settable via
  `xencode config set`; unknown layout names fall back to classic. The
  config dir honors `XCODE_CONFIG_DIR`
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
- TUI header reads ` ✦ xencode [layout] ⎇branch model` with the focused
  panel name on the right; parts drop out at 72/60/40 columns. Panel
  borders all route through one `panel_block` (rounded-corner aware), body
  pane titles drop their emoji under 30 columns, popups that would collapse
  below 16×6 grow to 80 % of the screen, and toasts no longer overlay the
  chat input on short terminals (Milestone H)
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