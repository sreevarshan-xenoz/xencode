# Official skills

This directory is a ready-to-use skills root in the format Xencode already
loads: one subdirectory per skill, each holding a `SKILL.md` with `name` and
`description` frontmatter followed by instructions. Only the name and the
one-line description are added to the system prompt; the body is read on demand
with the read-only `load_skill` tool. See `README.md` at the repository root for
how the skill system works.

## Included skills

| Name | When to use it |
|---|---|
| `rust-refactor` | Restructure or simplify Rust code in small compiler-verified steps |
| `write-tests` | Add real regression tests for code you changed |
| `security-pass` | Review a diff for input, auth, secret and path weaknesses |
| `commit-message` | Suggest a plain-English commit message from the real staged diff |
| `explain-diff` | Explain staged, unstaged or committed changes in plain terms |
| `find-dead-code` | Find unused Rust code and prove it is unused before deleting |
| `fix-clippy-warnings` | Fix Clippy warnings at the root, then re-run checks |
| `write-docs` | Update the manuals for a user-facing change |

## Install

Xencode scans two roots for skills: the settings directory (or `$XCODE_SKILLS_DIR`)
and `<workspace>/.xencode/skills`. A project skill replaces a user skill of the
same name. To make these official skills available, do one of the following:

- Point the environment variable at this directory, for example
  `XCODE_SKILLS_DIR=/path/to/xencode/skills xencode tui`. The variable is the
  supported override for portable installs and tests.
- Copy the individual skill subdirectories into your settings skills directory
  (`~/.config/xencode/skills` on a fresh install, or `~/.xencode/skills` on the
  older layout), or into `<workspace>/.xencode/skills` for one project only.

Then start (or reload with `/skills reload`) the session. `/skills` reports what
loaded and what was refused.

These are instructions for the model, not a feature of the binary: using them
requires a working local model that can call the `load_skill` tool. No skill is
run by Xencode itself. See `AGENTS.md` for the repository's skill format and the
rules these instructions enforce.
