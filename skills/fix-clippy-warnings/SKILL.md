---
name: fix-clippy-warnings
description: Fix Clippy warnings in a Rust crate. Use when a build shows warnings, before a commit, or when asked to clean up a crate's lint output.
allowed-tools:
  - read_file
  - search_files
  - edit_file
  - run_command
---
# Fix Clippy warnings

Fix the cause of each warning rather than silencing it. A broad `#[allow(...)]`
or a changed lint level is not a fix unless the warning is genuinely a false
positive, in which case say why in the report.

## Steps

1. From `rust/`, run `cargo clippy -p <crate> --all-targets -- -D warnings` for the
   crate in scope and capture the full warning list. Work from the real output;
   do not assume which lints will fire.
2. Group the warnings by file and kind (`needless_borrow`, `clone_on_copy`,
   `unnecessary_unwrap`, `dead_code`, and so on). Read the code with `read_file`
   before editing so the fix keeps behaviour and error handling intact.
3. Apply the smallest idiomatic change for each warning. Prefer clearer types,
   tighter scopes and removing redundant work over adding attributes. If a
   warning points at real dead code, verify with a search before removing it.
4. Re-run the same Clippy command until it is clean, then run
   `cargo fmt --all -- --check` and `cargo test -p <crate> -j 4`. A lint fix that
   breaks a test is not complete.
5. Report the count of warnings before and after, quote the final clean output,
   and list any warning you deliberately left with the reason. Never report a
   clean run you did not actually see.
