---
name: explain-diff
description: Explain a git diff in plain terms. Use when asked what a change does, to review someone else's work, or to understand uncommitted or staged edits before continuing.
allowed-tools:
  - read_file
  - list_dir
  - run_command
---
# Explain a diff

Turn the actual changes on disk into a short, accurate explanation. This is a
read-only task: do not edit, stage or commit anything while explaining.

## Steps

1. Find the changes with `git status --short`, then `git diff` for unstaged work
   and `git diff --cached` for staged work. If both are empty, say there are no
   local changes and stop.
2. If the user named a commit or branch, use `git show <ref>` or
   `git diff <base>..<head>` instead, and quote the ref you inspected.
3. Group the output by file. For each file, state what changed and why it matters:
   a new function, a changed condition, a moved test, a config or docs edit.
   Read the surrounding code with `read_file` when a hunk alone is ambiguous.
4. Separate behaviour changes from refactoring, and call out anything risky:
   a changed public API, a new dependency, a permission change, a migration or a
   file that was deleted. Say when a change is only formatting.
5. Close with a one-sentence summary of the whole change and the verification you
   saw in the diff (new tests, updated checks). If you cannot tell why something
   was changed from the diff alone, say so rather than guessing.
