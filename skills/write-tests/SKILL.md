---
name: write-tests
description: Add real tests for code you changed. Use when a fix or feature needs regression coverage, or when asked to raise coverage for a Rust crate.
allowed-tools:
  - read_file
  - search_files
  - write_file
  - edit_file
  - run_command
---
# Write tests that actually test

Tests in this repository must exercise real behaviour — real files, real sockets,
real git repositories, real provider calls where the protocol is the subject.
A test that replaces the thing under test with a canned answer proves nothing and
is not acceptable (see the no-mocks rule in `AGENTS.md`).

## Steps

1. Read the code under test and its existing tests in the same crate first. Match
   the surrounding style: most unit tests live in a `#[cfg(test)] mod tests`, and
   end-to-end tests live in the crate's `tests/` directory.
2. Write down the behaviour to pin down in one sentence. A good test name says
   what must happen, not what the function is called.
3. For a bug fix, first prove the test fails on the old behaviour, then apply the
   fix and watch it pass. For new code, make the assertion strong enough that an
   empty or stubbed implementation would fail it.
4. Use real inputs. For file work, use `tempfile::tempdir()` and remove it when the
   test finishes. For network code, use a real local listener. Never seed a result
   the implementation should compute.
5. Keep credentials out of fixtures. A made-up value must be outside a real
   vendor's key signature — the secret-fixture rule in `AGENTS.md` explains the
   shapes scanners flag and gives safe alternatives.
6. From `rust/`, run `cargo test -p <crate> -j 4` and quote the result, including
   the test count. Run `cargo fmt --all -- --check` on the changed files before
   finishing. If a test needs a service that is not available here, report it as
   blocked instead of inventing a passing result.
