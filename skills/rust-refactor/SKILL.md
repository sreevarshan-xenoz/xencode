---
name: rust-refactor
description: Refactor Rust code in small compiler-verified steps. Use when asked to restructure, simplify, extract or rename code in the Rust workspace while keeping behaviour identical.
allowed-tools:
  - read_file
  - list_dir
  - search_files
  - edit_file
  - rename
  - run_command
---
# Refactor Rust code safely

The product is Rust and lives in `rust/crates/*`. A refactor keeps behaviour
identical; if the task would change what the code does, treat that as a separate
change and say so before starting.

## Steps

1. Read `AGENTS.md` at the repository root before editing. It requires Rust-only
   changes, real implementations rather than mocks, one atomic commit per plan
   item, and a plain-English commit message that names the plan ID.
2. Locate the code and every reference to it. Use `read_file` and `list_dir` to
   understand the surrounding module, `search_files` (and `find_refs` / `callers`
   when available) to find all uses. Note anything public or cross-crate: removing
   or renaming it changes the crate's API.
3. Decide the smallest edit that moves the code one step toward the target. Prefer
   several small, compilable edits over one large rewrite. Keep existing names,
   error handling and ordering unless the task explicitly changes them.
4. After each edit, from `rust/` run `cargo fmt --all -- --check` and then
   `cargo test -p <crate> -j 4` for the crate you touched. Fix the code, not the
   test, and do not weaken an assertion just to make it pass.
5. When the crate is green, run `cargo clippy -p <crate> --all-targets -- -D warnings`
   to catch the warnings a refactor commonly introduces.
6. Before handing back, state what moved and why, list the files changed, and
   quote the test and check output. If a check could not run here, say which one
   and why rather than claiming it passed.
