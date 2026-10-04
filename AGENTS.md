# Xencode Agent Guidelines & Project Directives

## ⚠️ Critical Architecture Rule: Rust-Only

- **The product is Rust**: everything lives under the `rust/` workspace (`rust/crates/*`). The legacy Python stack was deleted — do **NOT** reintroduce Python code, packaging, or Python-based tooling (pip/pytest/ruff/bandit/PyInstaller).
- **Rust First**: All new development, bug fixes, features, TUI improvements, model provider integrations, settings, and CLI tools MUST be written in Rust under the `rust/` directory (`rust/crates/*`).
- When inspecting or fixing functionality (such as model selection, settings panel, Ollama integration, etc.), always target the Rust implementation (`rust/crates/xencode-tui-rs`, `rust/crates/xencode-models-rs`, `rust/crates/xencode-providers-rs`, `rust/crates/xencode-config-rs`, etc.).

## 🚫 No Mocks Rule: Real Implementations Only

- **No mocks anywhere** — not in product code, not in tests, not as scaffolding for unfinished features. Every implementation must be real and do what its name claims: actual file/Network/process/system behavior, actual provider calls, actual data. Stubs, fake data sources, canned responses, and "placeholder until we wire it up" shortcuts are forbidden.
- **Verification means it actually ran.** A done-when claim must come from watching the real behavior happen (live runs, real output, quoted numbers), never from a mock passing. If something can only be tested against a service or device that isn't here, say so and leave the item unchecked — do not fake it green.
- When a plan item's implementation would require a mock to finish, that item is **blocked, not done** — surface it to the user instead of papering over it.

## 🔄 Commit Rule: Atomic Git Commits

- **Commit Each Change Before Doing Next Changes**: Every logical task, feature, or bug fix MUST be committed to git immediately upon completion and verification before proceeding to the next change. Never accumulate multiple unrelated changes in the working tree without committing each step first.
- **The unit is the plan item, not the wave.** Work recorded in `NEXT_PLAN_TASKS.md` is ordered into fifteen dependency waves (§Milestone R), and a wave is a *set*, not a commit boundary. One plan item (an ID such as `SE-1`, `CI-2`, `W3`) = one change = **one commit**, made as soon as that item's own done-when is met — before starting the next item, even inside the same wave.
- **Name the IDs, always, in English.** Every commit message MUST name the `NEXT_PLAN_TASKS.md` ID(s) it completes, in the subject line — no exceptions. If two IDs are genuinely one change (e.g. `SE-1`'s `chmod` with `DB-1`'s atomic write), implement them together and **name both IDs in that one commit message**. A wave counts as done only when every non-parked ID in it can be traced to a commit that names it, checked against §R-1's tables rather than by eye.
- **Commit messages are user-facing product history and MUST be plain English.** Every commit message — subject *and* body — must be fully readable to a developer who has never seen an agent session in this repo. This rule has **no exceptions**: it applies to every commit without needing to be told, and it overrides any commit-message or response-language setting elsewhere in this file, in `~/.qoder/AGENTS.md`, or in a language detected from the conversation. Specifically forbidden in a commit message: bare plan/task IDs used as vocabulary (`QO-2`, `K-3`, §R-1 style references) where the thing they mean is not also stated in words; internal shorthand and coinages used as if they were self-explanatory (`pseudo-docs`, `telemetry-only`, `summarise`, `hard-cut`, `cadence`, `hard-refuse`, `hard-suppressed`, `gate`, `apply the fix`, `this fix`, `the fix`, `that check`, `proposal row 90`, `entry-point docs`, `core docs`, `top-up`, `seeded top-up`, `count-only`, `sweep`, `handoff`, `handoff doc`, `doc-drift`, `flake`, `flaky`, `blend`, `fixture`, `record` (as a noun), `replay` (as a noun), verbose run logs pasted in as evidence, and tool/JSON dumps like `VERDICTS_JSON` / `LOGS_JSONL` used as if they were narrative words. Also forbidden: the assistant's own fluent English mistakes used as if they were established terms — *verbalise*, *summarise*, *pseudo-docs* are examples of this, not a whitelist to check off. State what was wrong and what now happens, in words a stranger would understand; a correctly quoted code symbol or file name is fine and encouraged. If a message is written in a language other than the one the reply is in, it is still required to be plain and fully named in that language.

## 🚀 Push Rule: Do Not Push — Commits Stay Local

- **Never push.** Do NOT run `git push` unless the user explicitly asks for it in the current session. Commits stay on the local machine; the user pushes manually when ready.
- **No push verification needed.** Since nothing is pushed, there is nothing to verify against the remote. Never force-push unless the user explicitly asked for it.

## 📚 Docs Rule: Periodic Documentation Updates
- **Docs ride with features**: every user-facing change (new command, flag, TUI command, behavior change, install/infra change) MUST update the affected manuals in the same pass — `README.md`, `QUICK_START.md`, `CLI_GUIDE.md` as applicable — plus `NEXT_PLAN_TASKS.md` (check off finished items, fix stale counts) and `CHANGELOG.md` (Unreleased entries).
- **No fiction in manuals**: document only subcommands, flags, and behavior verified against the implementation (`--help` output, enum definitions, live runs). Never invent commands.
- **Counts stay current**: crate/test counts in `README.md` and `NEXT_PLAN_TASKS.md` must match `cargo test --workspace` at the time of the docs commit.
- **Sweep every few features**: after every 2–3 feature commits (or weekly, whichever comes first), re-check the entry-point docs for drift and correct everything in one docs commit.

## 🔐 Secret Fixture Rule: No Scanner-Flaggable Credentials In The Repo

- **The repo is public open source, and secret scanners (GitGuardian "Internal secret detection") fire on credential *shapes*, not on intent.** A test fixture that matches a real vendor's key pattern is flagged as an incident even when it is a made-up value. That noise hides genuine leaks, so credential-shaped strings in this repo are held to a stricter bar than "it isn't a real key".
- **Never** put a real/original credential anywhere in git — not in code, tests, docs, `CHANGELOG.md`, or a commit message. Real keys live only in `~/.config/xencode/config.json` / environment variables, outside the repo.
- **A placeholder must be unmistakably fake AND outside a vendor's published signature.** The credential values that appear in vendor *documentation* (the AWS access-key and secret-key examples, Google's fixed-length `AIzaSy…` form) and exact-length synthetic keys (`ghp_` + 36 base62, `sk-proj-` + long base62, `xoxb-<digits>-…`) are **forbidden** — they are precisely what scanners match. Prefer an obviously-fake token body that still trips **our own** detectors (`xencode-context-rs/src/trace.rs`: `secret_spans` / `redact_secrets`) but sits off the real signature: a wrong length (e.g. `AKIA` + a body that is not the vendor's fixed 16 characters), or hyphenated word bodies like `sk-FAKE-NOT-A-REAL-TEST-KEY`, `AIzaNOTREALKEYNOTREALKEY…`, `ghp_FAKE_NOT_A_REAL_TEST_KEY`, `xoxb-FAKE-NOT-A-REAL-TEST`. Do not write the vendor example values into this file either — quoting them here is itself a scanner hit.
- **Keep each detector's own test meaningful.** Do not over-fake a fixture so far that our regex stops matching — the whole point of these tests is proving a credential shape gets redacted. After changing any fixture, re-run the relevant crate's tests, not just the workspace build.


