---
name: security-pass
description: Review a change for security problems. Use before finishing a diff that touches input handling, authentication, secrets, shell commands, network code or file paths.
allowed-tools:
  - read_file
  - list_dir
  - search_files
  - run_command
---
# Run a security pass

Look for concrete weaknesses in the change in front of you, then verify each
finding with a real command. Report what you checked and what each check
returned; a clean report must say which checks actually ran.

## Steps

1. Read `AGENTS.md`, especially the secret-fixture rule. The repository is public
   and secret scanners match credential shapes, so no real or vendor-shaped
   credential may appear in code, tests, docs or commit messages.
2. Get the change with `git status` and `git diff` (add `--cached` for staged
   work). Review every hunk in the areas below:
   - Untrusted input: is it validated before use in a path, command, query or
     buffer? Sizes and lengths must be bounded.
   - Paths and commands: does a caller-supplied path stay inside the workspace,
     and can a command argument inject a shell?
   - Authentication and authorization: is every protected route or tool gated,
     and are failures refused rather than silently allowed?
   - Secrets: are values redacted in logs and traces, and are rate limits and
     timeouts applied to login and remote calls?
3. Run the repository's scanner for the touched path:
   `xencode analyze <path>`. It is a pattern-based OWASP Top 10 check; report its
   findings as findings, not as proof that the code is safe. If a deeper check
   needs `ast-grep` or another binary, say so when it is missing.
4. Search for credential patterns with `search_files` and check any hit against
   the safe-fixture examples in `AGENTS.md`. Use `cargo clippy -p <crate>
   --all-targets -- -D warnings` for the crate you touched.
5. Fix what you find with a real change and re-run the same checks. For each item,
   state the risk, the fix and the command output that confirms it. If something
   cannot be checked here, leave it listed as unchecked rather than calling the
   code secure.
