---
name: find-dead-code
description: Find unused Rust code and report it safely. Use when asked to locate dead functions, types, dependencies or files before deciding what to delete.
allowed-tools:
  - read_file
  - list_dir
  - search_files
  - find_refs
  - callers
  - run_command
---
# Find dead code

Find candidates for removal and prove they are unused before recommending any
deletion. Report findings; remove code only after the references and public API
have been checked and the user has agreed.

## Steps

1. From `rust/`, run `cargo clippy -p <crate> --all-targets -- -D warnings` for the
   crate in scope. Warnings such as `dead_code` and `unused_imports` are the
   compiler's own evidence and are the strongest candidates.
2. For every candidate, search the whole workspace with `search_files` for its
   name, and use `find_refs` / `callers` when available. Count real references,
   not the declaration itself and not matches inside comments or docs.
3. Check whether the item is public and part of another crate's API. A `pub`
   symbol may be used by a crate you are not currently building, by a test, or
   by an external consumer; `cargo test --workspace` is a better check than a
   single crate here.
4. Look for whole files and dependencies too: a module nobody imports, a dev
   dependency nobody mentions, a feature flag with no users. Confirm each with a
   search before listing it.
5. Present a table of candidate, file, evidence (which commands and searches)
   and a risk note. Recommend one removal at a time with the commands that would
   verify it, and never delete on a guess. If a check could not run, mark the
   candidate as unconfirmed.
