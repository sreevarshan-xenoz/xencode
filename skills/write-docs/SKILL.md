---
name: write-docs
description: Update the manuals for a user-facing change. Use when a command, flag, behavior or install step changes, or when docs have drifted from the code.
allowed-tools:
  - read_file
  - search_files
  - write_file
  - edit_file
  - run_command
---
# Update docs without writing fiction

`AGENTS.md` requires user-facing changes to ride with their manuals. Document
only behaviour you verified against the implementation; never invent a command,
flag or output.

## Steps

1. Read `AGENTS.md` for the docs rule and the plain-language requirements. Identify
   which manuals the change touches: `README.md`, `QUICK_START.md`, `CLI_GUIDE.md`
   as applicable, plus `NEXT_PLAN_TASKS.md` (check off the plan item and fix stale
   counts) and `CHANGELOG.md` (an Unreleased entry naming the plan ID).
2. Verify the surface before describing it. For CLI changes, run the built
   binary's `--help` and read the relevant enum or struct in
   `rust/crates/xencode-cli/src/cli.rs`. For behaviour changes, watch a real run
   or read the code path that produces the output.
3. Make the edit in the same pass as the code. Keep counts (crates, tests)
   aligned with a real `cargo test --workspace` run, and keep commit messages
   plain English — full words, no internal shorthand.
4. Re-read the surrounding section after editing so the new text agrees with the
   paragraphs around it. Remove or correct any stale claim you find, not just the
   one line you came to change.
5. Check the result with `git diff` and quote the commands whose output you
   verified. If a claim depends on a service unavailable here (for example a
   downloadable release or a GPU run), state plainly that it is unverified here
   instead of describing it as working.
